use wgpu::util::DeviceExt;

use crate::Engine;

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

        self.prefix_sum_inner(&storage_buffer);

        self.map_buffer(&storage_buffer).await
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

        let bufs = [buf, &next_buffer];

        self.dispatch_psum_kernel(&bufs, "psum1", 0);

        if input_len > 256 {
            self.prefix_sum_inner(&next_buffer);
            self.dispatch_psum_kernel(&bufs, "psum2", 1);
        }
    }

    fn dispatch_psum_kernel(&self, bufs: &[&wgpu::Buffer], kernel: &str, starting_offset: u32) {
        const MAX_WORKGROUPS: u32 = 65535;
        let total_wg_count = (bufs[0].size() / 4).div_ceil(256) as u32 - starting_offset;

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
                                has_dynamic_offset: true,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: true,
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

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: self.kernels.get(kernel).unwrap(),
                entry_point: "main",
            });

        let wg_remainder = total_wg_count % MAX_WORKGROUPS;
        let mut buf1_size_remainder = bufs[0].size() % (MAX_WORKGROUPS as u64 * 256 * 4);
        let buf2_size_remainder = buf1_size_remainder.div_ceil(256).max(4);
        buf1_size_remainder -= starting_offset as u64 * 256 * 4;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: bufs[0],
                        offset: starting_offset as u64 * 256 * 4,
                        size: buf1_size_remainder.try_into().ok(),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: bufs[1],
                        offset: 0,
                        size: buf2_size_remainder.try_into().ok(),
                    }),
                },
            ],
        });

        let mut bind_group_max_dispatch = None;
        let dispatch_count = total_wg_count.div_ceil(MAX_WORKGROUPS);

        if dispatch_count > 1 {
            bind_group_max_dispatch =
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: bufs[0],
                                offset: starting_offset as u64 * 256 * 4,
                                size: (MAX_WORKGROUPS as u64 * 256 * 4).try_into().ok(),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: bufs[1],
                                offset: 0,
                                size: (MAX_WORKGROUPS as u64 * 4).try_into().ok(),
                            }),
                        },
                    ],
                }));
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());

        for dispatch_i in 0..dispatch_count {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker(&format!("{} dispatch", kernel));
            cpass.set_pipeline(&pipeline);
            let offsets = [
                256 * 4 * dispatch_i * MAX_WORKGROUPS,
                256 * 4 * dispatch_i * (MAX_WORKGROUPS / 256),
            ];
            
            if dispatch_i == dispatch_count - 1 {
                cpass.set_bind_group(0, &bind_group, &offsets);
                cpass.dispatch_workgroups(wg_remainder, 1, 1);
            } else {
                cpass.set_bind_group(0, bind_group_max_dispatch.as_ref().unwrap(), &offsets);
                cpass.dispatch_workgroups(MAX_WORKGROUPS, 1, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tests::assert_slices_eq;

    pub(crate) fn prefix_sum_cpu(input: &[u32]) -> Vec<u32> {
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

        assert_slices_eq(&engine.prefix_sum(&[]).await?, &[]);
        assert_slices_eq(&engine.prefix_sum(&[3]).await?, &[3]);
        let input = vec![0; 1 << 20];
        assert_slices_eq(&engine.prefix_sum(&input).await?, &input);

        Ok(())
    }

    #[tokio::test]
    async fn short_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        for n in 1..=256 {
            let input: Vec<u32> = (1..=n).collect();
            let expected: Vec<u32> = prefix_sum_cpu(&input);
            let result = engine.prefix_sum(&input).await?;

            assert_slices_eq(&result, &expected);
        }

        Ok(())
    }

    #[tokio::test]
    async fn long_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        let input: Vec<u32> = (1..=(1u32 << 11)).collect();
        let expected: Vec<u32> = prefix_sum_cpu(&input);
        let result = engine.prefix_sum(&input).await?;

        assert_slices_eq(&result, &expected);

        Ok(())
    }

    #[tokio::test]
    async fn very_long_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        let input: Vec<u32> = (1..=16u32).cycle().take(1 << 23).collect();
        assert_slices_eq(&engine.prefix_sum(&input).await?, &prefix_sum_cpu(&input));

        Ok(())
    }

    #[ignore]
    #[tokio::test]
    async fn very_very_long_sum_works() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        let input: Vec<u32> = (1..=16u32).cycle().take(1 << 25).collect();
        let expected: Vec<u32> = prefix_sum_cpu(&input);
        let result = engine.prefix_sum(&input).await?;

        assert_slices_eq(&result, &expected);

        Ok(())
    }
}
