mod prefix_sum;
mod fenns;

use std::{borrow::Cow, collections::HashMap};

use anyhow::Context;

#[repr(C, align(16))]
#[derive(Copy, Clone, PartialEq, PartialOrd, bytemuck::Pod, bytemuck::Zeroable)]
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

impl std::fmt::Debug for Vec3A {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec3A({}, {}, {})", self.x, self.y, self.z)
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
        
        kernels.insert(
            "fenns_sort2".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/fenns_sort2.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/fenns_sort2.wgsl"))),
            }),
        );
        
        kernels.insert(
            "fenns_sort_shift".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/fenns_sort_shift.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("kernels/fenns_sort_shift.wgsl"))),
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
    use std::fmt::Write;

    pub(crate) fn print_slice_comparison<T: std::fmt::Debug>(pos: usize, name_a: &str, a: &[T], name_b: &str, b: &[T]) {
        assert_eq!(a.len(), b.len());

        let mut msg = String::new();

        let start = pos.saturating_sub(30);
        let end = (pos + 30).min(a.len() - 1);

        write!(
            msg,
            "{: <10} | {: <45} | {: <45}\n",
            "idx", name_a, name_b,
        ).unwrap();

        for display_i in start..end {
            write!(
                msg,
                "{: <10} | {: <45} | {: <45}\n",
                display_i, format!("{:?}", a[display_i]), format!("{:?}", b[display_i])
            ).unwrap();
        }

        print!("{}", msg);
    }

    pub(crate) fn assert_slices_eq<T: PartialEq + std::fmt::Debug>(left: &[T], right: &[T]) {
        assert_eq!(left.len(), right.len());
        for (i, (a, b)) in std::iter::zip(left.iter(), right.iter()).enumerate() {
            if a != b {
                print_slice_comparison(i, "left", left, "right", right);
                
                panic!("assertion failure: vec mismatch at index {}\n", i);
            }
        }
    }
}