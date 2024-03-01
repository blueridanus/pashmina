use crate::Engine;

impl Engine {
    pub fn fenns_sort1(&self, bufs: &[&wgpu::Buffer]) {
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: self.kernels.get("fenns_sort1").unwrap(),
                entry_point: "main",
            });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs[2].as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let len = bufs[1].size() / 16;

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_sort1 dispatch");
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(len.div_ceil(64) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3A;
    use crate::tests::assert_slices_eq;

    use std::iter::zip;

    use wgpu::util::DeviceExt;
    use rand::Rng;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};

    #[tokio::test]
    async fn check_fenns_sort1() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);

        const GRID_SIZE: usize = 18 * 18 * 18;
        let particle_counts: Vec<u32> = (0..GRID_SIZE).map(|_| rng.gen_range(1..=10)).collect();

        let mut particles: Vec<Vec3A> = vec![];

        for (i, n) in zip(0..GRID_SIZE, &particle_counts) {
            let x = i % 18;
            let y = (i / 18) % 18;
            let z = i / (18 * 18);

            for _ in 0..*n {
                particles.push(Vec3A::new(
                    x as f32 + rng.gen::<f32>(),
                    y as f32 + rng.gen::<f32>(),
                    z as f32 + rng.gen::<f32>(),
                ));
            }
        }

        println!("Particle count: {}", particles.len());
        assert_eq!(particles.len() as u32, particle_counts.iter().sum());

        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_sort1/buf0"),
                contents: bytemuck::cast_slice(&[1f32]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let particles_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_sort1/buf1"),
                contents: bytemuck::cast_slice(&particles),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_sort1/buf2"),
            size: 4 * GRID_SIZE as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        engine.fenns_sort1(&[&params_buf, &particles_buf, &count_buf]);
        let result = engine.map_buffer(&count_buf).await?;

        assert_slices_eq(&result, &particle_counts);

        Ok(())
    }
}
