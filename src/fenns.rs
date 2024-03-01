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

    pub fn fenns_sort_shift(&self, buf: &wgpu::Buffer) {
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: self.kernels.get("fenns_sort_shift").unwrap(),
                entry_point: "main",
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let len = buf.size() / 4;
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_sort_shuffle dispatch");
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(len.div_ceil(64) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn fenns_sort2(&self, bufs: &[&wgpu::Buffer]) {
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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
                module: self.kernels.get("fenns_sort2").unwrap(),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs[3].as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let len = bufs[1].size() / 16;

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_sort2 dispatch");
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
    use crate::tests::{assert_slices_eq, print_slice_comparison};
    use crate::Vec3A;

    use std::iter::zip;

    use rand::Rng;
    use rand_xoshiro::{rand_core::SeedableRng, Xoshiro256PlusPlus};
    use wgpu::util::DeviceExt;

    pub fn gen_particles(seed: u64, grid_dim: usize) -> (Vec<Vec3A>, Vec<u32>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

        let grid_size: usize = grid_dim * grid_dim * grid_dim;
        let particle_counts: Vec<u32> = (0..grid_size).map(|_| rng.gen_range(1..=10)).collect();

        let mut particles: Vec<Vec3A> = vec![];

        for (i, n) in zip(0..grid_size, &particle_counts) {
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

        (particles, particle_counts)
    }

    #[tokio::test]
    async fn check_fenns_sort1() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        const SEED: u64 = 0;
        const GRID_DIM: usize = 18;
        const GRID_SIZE: usize = GRID_DIM * GRID_DIM * GRID_DIM;
        let (particles, particle_counts) = gen_particles(SEED, GRID_DIM);

        println!("Particle count: {}", particles.len());
        assert_eq!(particles.len() as u32, particle_counts.iter().sum());

        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_sort1/buf0"),
                contents: bytemuck::cast_slice(&[1f32, 0.1f32]),
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
            size: 4 * GRID_SIZE as u64 * 2,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        engine.fenns_sort1(&[&params_buf, &particles_buf, &count_buf]);
        let result = engine.map_buffer(&count_buf).await?;

        assert_slices_eq(&result[0..GRID_SIZE], &particle_counts);

        Ok(())
    }

    
    #[tokio::test]
    async fn check_fenns_sort2() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        const SEED: u64 = 0;
        const GRID_DIM: usize = 18;
        const GRID_SIZE: usize = GRID_DIM * GRID_DIM * GRID_DIM;
        const CELL_SIZE: f32 = 1.0;
        const SEARCH_RADIUS: f32 = 0.1;
        let (particles, particle_counts) = gen_particles(SEED, GRID_DIM);

        println!("Particle count: {}", particles.len());
        assert_eq!(particles.len() as u32, particle_counts.iter().sum());

        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_sort2/buf0"),
                contents: bytemuck::cast_slice(&[CELL_SIZE, SEARCH_RADIUS]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let particles_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_sort2/buf1"),
                contents: bytemuck::cast_slice(&particles),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_sort2/buf2"),
            size: 4 * GRID_SIZE as u64 * 2,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let reordered_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_sort2/buf3"),
            size: particles.len() as u64 * 16,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        engine.fenns_sort1(&[&params_buf, &particles_buf, &count_buf]);
        engine.prefix_sum_inner(&count_buf);
        engine.fenns_sort_shift(&count_buf);
        
        let shifted: Vec<u32> = engine.map_buffer(&count_buf).await?;
        assert_eq!(shifted[particle_counts.len()], 0);
        assert_slices_eq(&shifted[..particle_counts.len()-1], &shifted[particle_counts.len()+1..]);

        engine.fenns_sort2(&[&params_buf, &particles_buf, &count_buf, &reordered_buf]);

        let reordered: Vec<Vec3A> = engine.map_buffer(&reordered_buf).await?;

        // print_slice_comparison(1282, "before", &particles, "after", &reordered);
        // panic!();
        
        let is_border_particle = |particle: Vec3A| {
            for coord in &[particle.x, particle.y, particle.z]{
                let cell_coord = (coord / CELL_SIZE).fract();
                if cell_coord < SEARCH_RADIUS || cell_coord > CELL_SIZE - SEARCH_RADIUS {
                    return true;
                }
            }
            return false;      
        };
        let mut i = 0;
        for count in particle_counts.into_iter() {
            let mut border_j = 0;
            let mut nonborder_j = 0;

            for j in 0..(count as usize) {
                if !particles[i..i+(count as usize)].iter().find(|&&x| reordered[i+j] == x).is_some() {
                    println!("Test error: expected to find particle in the same grid cell after reorder");
                    println!();
                    println!("Particle #{} (after reorder) is {:?}", i+j, reordered[i+j]);
                    println!("Should have been one of those (before reorder) at grid cell idx {}:", i);
                    for candidate_i in i..i+(count as usize) {
                        println!("  #{}: {:?}", candidate_i, particles[candidate_i]);
                    }
                    println!();
                    print_slice_comparison(i, "particles before", &particles, "after reorder", &reordered);
                    panic!();
                };

                let particle = particles[i+j];
                // TODO: check if reorder idxs are correct
                if is_border_particle(particle) {
                    border_j += 1;
                } else {
                    nonborder_j += 1;
                }
            }
            i += count as usize;
        }

        Ok(())
    }
}
