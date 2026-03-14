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
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut input_len = buf.size() / 4;
        let mut level_input_lens = Vec::new();
        let mut scratch = Vec::new();

        loop {
            level_input_lens.push(input_len);
            scratch.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("next buffer"),
                size: 4 * input_len.div_ceil(256),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));

            if input_len <= 256 {
                break;
            }

            input_len = input_len.div_ceil(256);
        }

        for level in 0..scratch.len() {
            let src = if level == 0 { buf } else { &scratch[level - 1] };
            let dst = &scratch[level];
            let bufs = [src, dst];
            self.dispatch_psum_kernel_encoded(&bufs, "psum1", 0, &mut encoder);
        }

        for level in (0..scratch.len()).rev() {
            if level_input_lens[level] <= 256 {
                continue;
            }

            let src = if level == 0 { buf } else { &scratch[level - 1] };
            let dst = &scratch[level];
            let bufs = [src, dst];
            self.dispatch_psum_kernel_encoded(&bufs, "psum2", 1, &mut encoder);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_psum_kernel_encoded(
        &self,
        bufs: &[&wgpu::Buffer],
        kernel: &str,
        starting_offset: u32,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        const MAX_WORKGROUPS: u32 = 65535;
        let total_wg_count = (bufs[0].size() / 4).div_ceil(256) as u32 - starting_offset;
        if total_wg_count == 0 {
            return;
        }

        let pipeline = match kernel {
            "psum1" => &self.pipelines.psum1,
            "psum2" => &self.pipelines.psum2,
            _ => unreachable!("unknown prefix-sum kernel"),
        };

        let dispatch_window_wg = total_wg_count.min(MAX_WORKGROUPS);
        let buf0_offset = starting_offset as u64 * 256 * 4;
        let buf0_size = (bufs[0].size() - buf0_offset).min(dispatch_window_wg as u64 * 256 * 4);
        let buf1_size = bufs[1].size().min((dispatch_window_wg as u64 * 4).max(4));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.psum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: bufs[0],
                        offset: buf0_offset,
                        size: buf0_size.try_into().ok(),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: bufs[1],
                        offset: 0,
                        size: buf1_size.try_into().ok(),
                    }),
                },
            ],
        });

        let dispatch_count = total_wg_count.div_ceil(MAX_WORKGROUPS);

        for dispatch_i in 0..dispatch_count {
            let dispatched_wg = dispatch_i * MAX_WORKGROUPS;
            let remaining_wg = total_wg_count - dispatched_wg;
            let wg_count = remaining_wg.min(MAX_WORKGROUPS);
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker(&format!("{} dispatch", kernel));
            cpass.set_pipeline(&pipeline);
            let offsets = [256 * 4 * dispatched_wg, 4 * dispatched_wg];
            cpass.set_bind_group(0, &bind_group, &offsets);
            cpass.dispatch_workgroups(wg_count, 1, 1);
        }
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
