use std::borrow::Cow;

use anyhow::Context;
use wgpu::{util::DeviceExt, Device, Queue};

async fn init_device() -> anyhow::Result<(Device, Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .context("Adapter initialization failed")?;

    let device = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits {
                    max_compute_invocations_per_workgroup: 512,
                    ..Default::default()
                },
            },
            None,
        ).await?;

    Ok(device)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (device, queue) = init_device()
        .await
        .context("Device initialization failed")?;

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    const INVOCATIONS: u64 = 4096;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer"),
        size: 16*INVOCATIONS,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("storage buffer"),
        contents: &[0u8; 16*INVOCATIONS as usize],
        usage: wgpu::BufferUsages::STORAGE
             | wgpu::BufferUsages::COPY_DST
             | wgpu::BufferUsages::COPY_SRC,
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    
    {
        let mut cpass = encoder.begin_compute_pass(&Default::default());
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute example");
        cpass.dispatch_workgroups(16, 16, 16);
    }

    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, 16*INVOCATIONS);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    receiver.recv_async().await??;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<[u32; 4]> = bytemuck::cast_slice(&data).to_vec();

    drop(data);
    staging_buffer.unmap();

    println!("Hello, wgpu!");

    for i in 0..50 {
        println!("x: {}\t y: {}\t z: {}", &result[i][0], &result[i][1], &result[i][2]);
    }

    Ok(())
}