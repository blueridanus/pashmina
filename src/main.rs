use wgpu::util::DeviceExt;

mod engine;
mod prefix_sum;

use engine::Engine;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = Engine::new().await?;

    let input: Vec<u32> = Vec::from_iter(1..=256 as u32);

    let staging_buffer = engine.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer"),
        size: 4 * input.len() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let storage_buffer = engine
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage buffer"),
            contents: bytemuck::cast_slice(input.as_slice()),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

    engine.prefix_sum_inner(&storage_buffer);

    let mut encoder = engine.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &staging_buffer,
        0,
        4 * input.len() as u64,
    );
    engine.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    engine
        .device
        .poll(wgpu::Maintain::wait())
        .panic_on_timeout();

    receiver.recv_async().await??;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

    drop(data);
    staging_buffer.unmap();

    for i in 0..input.len().min(50) {
        println!("{}: {}", i + 1, &result[i as usize]);
    }

    Ok(())
}
