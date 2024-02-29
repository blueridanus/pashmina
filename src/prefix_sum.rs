use wgpu::util::DeviceExt;

use crate::engine::Engine;

impl Engine {
    pub async fn prefix_sum(&self, input: &[u32]) -> anyhow::Result<Vec<u32>> {
        if input.len() <= 1 {
            return Ok(Vec::from(input));
        }

        let storage_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("storage buffer"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: 4 * input.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.prefix_sum_inner(&storage_buffer);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            &storage_buffer,
            0,
            &staging_buffer,
            0,
            4 * input.len() as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        receiver.recv_async().await??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    pub fn prefix_sum_inner(&self, buf: &wgpu::Buffer) {
        let input_len = buf.size() / 4;

        let next_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("next buffer"),
            size: 4 * (input_len).div_ceil(256),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: self.kernels.get("psum1").unwrap(),
                    entry_point: "main",
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: next_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("psum1 dispatch");
            cpass.dispatch_workgroups(input_len.div_ceil(256) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        if input_len > 256 {
            self.prefix_sum_inner(&next_buffer);

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: self.kernels.get("psum2").unwrap(),
                        entry_point: "main",
                    });

            let mut encoder = self.device.create_command_encoder(&Default::default());

            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.insert_debug_marker("psum2 dispatch");
                cpass.dispatch_workgroups((input_len.div_ceil(256) as u32) - 1, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefix_sum_cpu(input: &[u32]) -> Vec<u32> {
        input
            .iter()
            .scan(0u32, |acc, x| {
                *acc += x;
                Some(*acc)
            })
            .collect()
    }

    #[tokio::test]
    async fn trivial_input_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        assert_eq!(engine.prefix_sum(&[]).await?, vec![]);
        assert_eq!(engine.prefix_sum(&[3]).await?, vec![3]);

        let input = vec![0; 1 << 20];
        assert_eq!(engine.prefix_sum(&input).await?, input);

        Ok(())
    }

    #[tokio::test]
    async fn short_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        for n in 1..=256 {
            let input: Vec<u32> = (1..=n).collect();
            let expected: Vec<u32> = prefix_sum_cpu(&input);
            let result = engine.prefix_sum(&input).await?;

            assert_eq!(result, expected);
        }

        Ok(())
    }

    #[tokio::test]
    async fn long_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        let input: Vec<u32> = (1..=(1u32 << 11)).collect();
        let expected: Vec<u32> = prefix_sum_cpu(&input);
        let result = engine.prefix_sum(&input).await?;

        assert_eq!(result, expected);

        Ok(())
    }

    #[tokio::test]
    async fn very_long_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        let input: Vec<u32> = (1..=16u32).cycle().take(1 << 25).collect();
        let expected: Vec<u32> = prefix_sum_cpu(&input);
        let result = engine.prefix_sum(&input).await?;

        assert_eq!(result, expected);

        Ok(())
    }
}
