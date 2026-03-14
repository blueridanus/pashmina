use std::time::Instant;

use pashmina::{Engine, Vec3A};
use tracing_subscriber;
use wgpu::util::DeviceExt;

const GRID_DIM: usize = 18;
const GRID_SIZE: usize = GRID_DIM * GRID_DIM * GRID_DIM;
const CELL_SIZE: f32 = 1.0;
const SEARCH_RADIUS: f32 = 0.1;
const WARMUP_ITERS: usize = 5;
const MEASURE_ITERS: usize = 75;
const PARTICLE_COUNTS: [usize; 6] = [1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 19, 1 << 20];

fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9e3779b9);
    x = (x ^ (x >> 16)).wrapping_mul(0x85ebca6b);
    x = (x ^ (x >> 13)).wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

fn unit_open(seed: u32) -> f32 {
    let bits = splitmix32(seed) >> 8;
    bits as f32 / ((1u32 << 24) as f32)
}

fn gen_particles(n: usize) -> Vec<Vec3A> {
    let mut particles = Vec::with_capacity(n);

    for i in 0..n {
        let cell_idx = i % GRID_SIZE;
        let x = (cell_idx % GRID_DIM) as f32;
        let y = ((cell_idx / GRID_DIM) % GRID_DIM) as f32;
        let z = (cell_idx / (GRID_DIM * GRID_DIM)) as f32;
        let seed = i as u32;

        particles.push(Vec3A::new(
            x + unit_open(seed.wrapping_mul(3).wrapping_add(0)),
            y + unit_open(seed.wrapping_mul(3).wrapping_add(1)),
            z + unit_open(seed.wrapping_mul(3).wrapping_add(2)),
        ));
    }

    particles
}

fn median_ms(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if samples.is_empty() {
        return 0.0;
    }

    let mid = samples.len() / 2;
    if samples.len() % 2 == 0 {
        (samples[mid - 1] + samples[mid]) * 0.5
    } else {
        samples[mid]
    }
}

fn avg_ms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let engine = Engine::new().await?;

    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}",
        "particles", "prep_ms", "search_ms", "avg_ms"
    );

    for particle_count in PARTICLE_COUNTS {
        let particles = gen_particles(particle_count);
        let params_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bench/params"),
                contents: bytemuck::cast_slice(&[CELL_SIZE, SEARCH_RADIUS]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let particles_buf = engine
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bench/particles"),
                contents: bytemuck::cast_slice(&particles),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench/count"),
            size: 4 * GRID_SIZE as u64 * 2,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let reordered_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench/reordered"),
            size: particles.len() as u64 * 16,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let border_count_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench/border_count"),
            size: 4 * GRID_SIZE as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let neighbors_buf = engine.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bench/neighbors"),
            size: particles.len() as u64 * 4,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let prep_start = Instant::now();
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
        engine
            .device
            .poll(wgpu::Maintain::wait())
            .panic_on_timeout();
        let prep_ms = prep_start.elapsed().as_secs_f64() * 1000.0;

        for _ in 0..WARMUP_ITERS {
            engine.fenns_search(&[&params_buf, &reordered_buf, &count_buf, &neighbors_buf]);
            engine
                .device
                .poll(wgpu::Maintain::wait())
                .panic_on_timeout();
        }

        let mut samples = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let start = Instant::now();
            engine.fenns_search(&[&params_buf, &reordered_buf, &count_buf, &neighbors_buf]);
            engine
                .device
                .poll(wgpu::Maintain::wait())
                .panic_on_timeout();
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let avg = avg_ms(&samples);
        let med = median_ms(&mut samples);

        println!(
            "{:>10}  {:>10.3}  {:>10.3}  {:>10.3}",
            particle_count, prep_ms, med, avg
        );
    }

    Ok(())
}
