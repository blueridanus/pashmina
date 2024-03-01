mod prefix_sum;
mod fenns;

use std::{borrow::Cow, collections::HashMap};

use anyhow::Context;

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vec3A {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    _padding: [u8; 4],
}

impl Vec3A {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            _padding: [0u8; 4],
        }
    }
}

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub kernels: HashMap<String, wgpu::ShaderModule>,
}

impl Engine {
    pub async fn map_buffer<T: bytemuck::Pod>(&self, buf: &wgpu::Buffer) -> anyhow::Result<Vec<T>> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging buffer"),
            size: buf.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(
            buf,
            0,
            &staging_buffer,
            0,
            buf.size(),
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        receiver.recv_async().await??;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    pub async fn new() -> anyhow::Result<Self> {
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: Default::default(),
                },
                None,
            )
            .await?;

        let mut kernels = HashMap::new();

        kernels.insert(
            "psum1".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/psum1.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/psum1.wgsl"))),
            }),
        );

        kernels.insert(
            "psum2".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/psum2.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/psum2.wgsl"))),
            }),
        );

        kernels.insert(
            "fenns_sort1".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/fenns_sort1.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/fenns_sort1.wgsl"))),
            }),
        );

        Ok(Self {
            device,
            queue,
            kernels,
        })
    }
}

#[cfg(test)]
mod tests {
    pub(crate) fn assert_slices_eq(left: &[u32], right: &[u32]) {
        assert_eq!(left.len(), right.len());
        for (i, (a, b)) in std::iter::zip(left.iter(), right.iter()).enumerate() {
            if a != b {
                let mut error_msg = format!("assertion failure: vec mismatch at index {}\n", i);
                error_msg.push_str(&format!(
                    "{: <10} | {: <15} | {: <15}\n",
                    "index", "left", "right"
                ));
                let start = i.saturating_sub(10);
                let end = (i + 10).min(left.len() - 1);
                for display_i in start..end {
                    error_msg.push_str(&format!(
                        "{: <10} | {: <15} | {: <15}\n",
                        display_i, left[display_i], right[display_i]
                    ));
                }
                panic!("{}", error_msg);
            }
        }
    }
}