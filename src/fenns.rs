use crate::Engine;

impl Engine {
    const FENNS_WG_SIZE: u64 = 64;
    const FENNS_LINEAR_WG_SIZE: u64 = 256;

    pub fn fenns_sort1(&self, bufs: &[&wgpu::Buffer]) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.fenns_sort1_bind_group_layout,
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
            cpass.set_pipeline(&self.pipelines.fenns_sort1);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(len.div_ceil(Self::FENNS_WG_SIZE) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn fenns_sort_shift(&self, buf: &wgpu::Buffer) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.fenns_sort_shift_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let len = buf.size() / 8;
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_sort_shuffle dispatch");
            cpass.set_pipeline(&self.pipelines.fenns_sort_shift);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(len.div_ceil(Self::FENNS_LINEAR_WG_SIZE) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn fenns_sort2(&self, bufs: &[&wgpu::Buffer]) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.fenns_sort2_bind_group_layout,
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs[4].as_entire_binding(),
                },
            ],
        });

        let restore_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.fenns_sort_restore_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs[4].as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let len = bufs[1].size() / 16;
        let grid_len = bufs[4].size() / 4;

        encoder.clear_buffer(bufs[4], 0, None);

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_sort2 dispatch");
            cpass.set_pipeline(&self.pipelines.fenns_sort2);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(len.div_ceil(Self::FENNS_WG_SIZE) as u32, 1, 1);
        }

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_sort_restore dispatch");
            cpass.set_pipeline(&self.pipelines.fenns_sort_restore);
            cpass.set_bind_group(0, &restore_bind_group, &[]);
            cpass.dispatch_workgroups(grid_len.div_ceil(Self::FENNS_LINEAR_WG_SIZE) as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn fenns_search(&self, bufs: &[&wgpu::Buffer]) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipelines.fenns_search_bind_group_layout,
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
        let grid_len = bufs[2].size() / 8;

        encoder.clear_buffer(bufs[3], 0, None);

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.insert_debug_marker("fenns_search dispatch");
            cpass.set_pipeline(&self.pipelines.fenns_search);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(grid_len as u32, 1, 1);
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
        let unit_open = |rng: &mut Xoshiro256PlusPlus| rng.gen::<f32>() * 0.9999;

        let grid_size: usize = grid_dim * grid_dim * grid_dim;
        let particle_counts: Vec<u32> = (0..grid_size).map(|_| rng.gen_range(1..=10)).collect();

        let mut particles: Vec<Vec3A> = vec![];

        for (i, n) in zip(0..grid_size, &particle_counts) {
            let x = i % 18;
            let y = (i / 18) % 18;
            let z = i / (18 * 18);

            for _ in 0..*n {
                particles.push(Vec3A::new(
                    x as f32 + unit_open(&mut rng),
                    y as f32 + unit_open(&mut rng),
                    z as f32 + unit_open(&mut rng),
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

        for seed in 0..50 {
            println!("Seed: {}", seed);
            check_fenns_sort2_inner(&engine, seed).await?;
        }

        Ok(())
    }

    async fn check_fenns_sort2_inner(engine: &Engine, seed: u64) -> anyhow::Result<()> {
        const GRID_DIM: usize = 18;
        const GRID_SIZE: usize = GRID_DIM * GRID_DIM * GRID_DIM;
        const CELL_SIZE: f32 = 1.0;
        const SEARCH_RADIUS: f32 = 0.1;
        let (particles, particle_counts) = gen_particles(seed, GRID_DIM);

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

        let border_count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_sort2/buf4"),
            size: 4 * GRID_SIZE as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        engine.fenns_sort1(&[&params_buf, &particles_buf, &count_buf]);

        let counts: Vec<u32> = engine.map_buffer(&count_buf).await?;

        engine.prefix_sum_inner(&count_buf);

        let summed: Vec<u32> = engine.map_buffer(&count_buf).await?;
        let expected_sum = crate::prefix_sum::tests::prefix_sum_cpu(&counts);

        assert_slices_eq(&summed, &expected_sum);

        engine.fenns_sort_shift(&count_buf);

        let shifted: Vec<u32> = engine.map_buffer(&count_buf).await?;
        assert_eq!(shifted[particle_counts.len()], 0);
        assert_slices_eq(
            &shifted[..particle_counts.len() - 1],
            &shifted[particle_counts.len() + 1..],
        );

        engine.fenns_sort2(&[
            &params_buf,
            &particles_buf,
            &count_buf,
            &reordered_buf,
            &border_count_buf,
        ]);

        let reordered: Vec<Vec3A> = engine.map_buffer(&reordered_buf).await?;

        let original_zeros = particles
            .iter()
            .filter(|&&v| v == Vec3A::new(0.0, 0.0, 0.0))
            .count();
        let reordered_zeros = reordered
            .iter()
            .enumerate()
            .filter(|(_i, &v)| v == Vec3A::new(0.0, 0.0, 0.0));

        if reordered_zeros.clone().count() != original_zeros {
            panic!(
                "Reordering has zero particles: {:?}",
                reordered_zeros.collect::<Vec<(usize, _)>>()
            )
        }
        let mut i = 0;
        for count in particle_counts.into_iter() {
            for j in 0..(count as usize) {
                if !particles[i..i + (count as usize)]
                    .iter()
                    .find(|&&x| reordered[i + j] == x)
                    .is_some()
                {
                    println!(
                        "Test error: expected to find particle in the same grid cell after reorder"
                    );
                    println!();
                    println!(
                        "Particle #{} (after reorder) is {:?}",
                        i + j,
                        reordered[i + j]
                    );
                    println!(
                        "Should have been one of those (before reorder) at grid cell idx {}:",
                        i
                    );
                    for candidate_i in i..i + (count as usize) {
                        println!("  #{}: {:?}", candidate_i, particles[candidate_i]);
                    }
                    println!();
                    print_slice_comparison(
                        i,
                        "particles before",
                        &particles,
                        "after reorder",
                        &reordered,
                    );
                    panic!();
                };
            }
            i += count as usize;
        }

        Ok(())
    }

    fn cpu_neighbor_counts(input: &[Vec3A], search_radius: f32) -> Vec<u32> {
        let radius_sq = search_radius * search_radius;

        input
            .iter()
            .enumerate()
            .map(|(i, a)| {
                input
                    .iter()
                    .enumerate()
                    .filter(|(j, b)| {
                        if i == *j {
                            return false;
                        }

                        let dx = a.x - b.x;
                        let dy = a.y - b.y;
                        let dz = a.z - b.z;
                        dx * dx + dy * dy + dz * dz <= radius_sq
                    })
                    .count() as u32
            })
            .collect()
    }

    #[tokio::test]
    async fn check_fenns_search() -> anyhow::Result<()> {
        let engine = Engine::new().await?;

        const GRID_DIM: usize = 18;
        const GRID_SIZE: usize = GRID_DIM * GRID_DIM * GRID_DIM;
        const CELL_SIZE: f32 = 1.0;
        const SEARCH_RADIUS: f32 = 0.2;

        let particles = vec![
            Vec3A::new(0.10, 0.10, 0.10),
            Vec3A::new(0.25, 0.10, 0.10),
            Vec3A::new(0.95, 0.10, 0.10),
            Vec3A::new(1.05, 0.10, 0.10),
            Vec3A::new(1.80, 0.10, 0.10),
            Vec3A::new(0.10, 0.95, 0.10),
            Vec3A::new(0.10, 1.05, 0.10),
            Vec3A::new(0.10, 0.10, 0.95),
            Vec3A::new(0.10, 0.10, 1.05),
        ];

        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_search/buf0"),
                contents: bytemuck::cast_slice(&[CELL_SIZE, SEARCH_RADIUS]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let particles_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fenns_search/buf1"),
                contents: bytemuck::cast_slice(&particles),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_search/buf2"),
            size: 4 * GRID_SIZE as u64 * 2,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let reordered_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_search/buf3"),
            size: particles.len() as u64 * 16,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let border_count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_search/buf4"),
            size: 4 * GRID_SIZE as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let neighbors_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fenns_search/buf5"),
            size: particles.len() as u64 * 4,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        engine.fenns_sort1(&[&params_buf, &particles_buf, &count_buf]);
        engine.prefix_sum_inner(&count_buf);
        engine.fenns_sort_shift(&count_buf);
        engine.fenns_sort2(&[
            &params_buf,
            &particles_buf,
            &count_buf,
            &reordered_buf,
            &border_count_buf,
        ]);
        engine.fenns_search(&[&params_buf, &reordered_buf, &count_buf, &neighbors_buf]);

        let reordered: Vec<Vec3A> = engine.map_buffer(&reordered_buf).await?;
        let result: Vec<u32> = engine.map_buffer(&neighbors_buf).await?;
        let expected = cpu_neighbor_counts(&reordered, SEARCH_RADIUS);

        assert_slices_eq(&result, &expected);

        Ok(())
    }
}
