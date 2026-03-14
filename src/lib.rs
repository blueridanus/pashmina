mod fenns;
mod prefix_sum;

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
    pub pipelines: Pipelines,
}

pub struct Pipelines {
    pub psum_bind_group_layout: wgpu::BindGroupLayout,
    pub psum1: wgpu::ComputePipeline,
    pub psum2: wgpu::ComputePipeline,
    pub fenns_sort1_bind_group_layout: wgpu::BindGroupLayout,
    pub fenns_sort1: wgpu::ComputePipeline,
    pub fenns_sort2_bind_group_layout: wgpu::BindGroupLayout,
    pub fenns_sort2: wgpu::ComputePipeline,
    pub fenns_sort_shift_bind_group_layout: wgpu::BindGroupLayout,
    pub fenns_sort_shift: wgpu::ComputePipeline,
    pub fenns_search_bind_group_layout: wgpu::BindGroupLayout,
    pub fenns_search: wgpu::ComputePipeline,
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
        encoder.copy_buffer_to_buffer(buf, 0, &staging_buffer, 0, buf.size());

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

        #[cfg(test)]
        println!("{:?}\n", adapter.get_info());

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
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "kernels/fenns_sort1.wgsl"
                ))),
            }),
        );

        kernels.insert(
            "fenns_sort2".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/fenns_sort2.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "kernels/fenns_sort2.wgsl"
                ))),
            }),
        );

        kernels.insert(
            "fenns_sort_shift".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/fenns_sort_shift.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "kernels/fenns_sort_shift.wgsl"
                ))),
            }),
        );

        kernels.insert(
            "fenns_search".into(),
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kernels/fenns_search.wgsl"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "kernels/fenns_search.wgsl"
                ))),
            }),
        );

        let psum_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layouts/psum"),
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
        let psum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipelines/psum"),
            bind_group_layouts: &[&psum_bind_group_layout],
            push_constant_ranges: &[],
        });
        let psum1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipelines/psum1"),
            layout: Some(&psum_pipeline_layout),
            module: kernels.get("psum1").unwrap(),
            entry_point: "main",
        });
        let psum2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipelines/psum2"),
            layout: Some(&psum_pipeline_layout),
            module: kernels.get("psum2").unwrap(),
            entry_point: "main",
        });

        let fenns_sort1_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layouts/fenns_sort1"),
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
        let fenns_sort1_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipelines/fenns_sort1"),
                bind_group_layouts: &[&fenns_sort1_bind_group_layout],
                push_constant_ranges: &[],
            });
        let fenns_sort1 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipelines/fenns_sort1"),
            layout: Some(&fenns_sort1_pipeline_layout),
            module: kernels.get("fenns_sort1").unwrap(),
            entry_point: "main",
        });

        let fenns_sort2_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layouts/fenns_sort2"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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
        let fenns_sort2_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipelines/fenns_sort2"),
                bind_group_layouts: &[&fenns_sort2_bind_group_layout],
                push_constant_ranges: &[],
            });
        let fenns_sort2 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipelines/fenns_sort2"),
            layout: Some(&fenns_sort2_pipeline_layout),
            module: kernels.get("fenns_sort2").unwrap(),
            entry_point: "main",
        });

        let fenns_sort_shift = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipelines/fenns_sort_shift"),
            layout: None,
            module: kernels.get("fenns_sort_shift").unwrap(),
            entry_point: "main",
        });
        let fenns_sort_shift_bind_group_layout = fenns_sort_shift.get_bind_group_layout(0);

        let fenns_search_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("layouts/fenns_search"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let fenns_search_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pipelines/fenns_search"),
                bind_group_layouts: &[&fenns_search_bind_group_layout],
                push_constant_ranges: &[],
            });
        let fenns_search = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipelines/fenns_search"),
            layout: Some(&fenns_search_pipeline_layout),
            module: kernels.get("fenns_search").unwrap(),
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            kernels,
            pipelines: Pipelines {
                psum_bind_group_layout,
                psum1,
                psum2,
                fenns_sort1_bind_group_layout,
                fenns_sort1,
                fenns_sort2_bind_group_layout,
                fenns_sort2,
                fenns_sort_shift_bind_group_layout,
                fenns_sort_shift,
                fenns_search_bind_group_layout,
                fenns_search,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    pub(crate) fn print_slice_comparison<T: std::fmt::Debug>(
        pos: usize,
        name_a: &str,
        a: &[T],
        name_b: &str,
        b: &[T],
    ) {
        assert_eq!(a.len(), b.len());

        let mut msg = String::new();

        let start = pos.saturating_sub(30);
        let end = (pos + 30).min(a.len());

        write!(msg, "{: <10} | {: <45} | {: <45}\n", "idx", name_a, name_b,).unwrap();

        for display_i in start..end {
            write!(
                msg,
                "{: <10} | {: <45} | {: <45}\n",
                display_i,
                format!("{:?}", a[display_i]),
                format!("{:?}", b[display_i])
            )
            .unwrap();
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
