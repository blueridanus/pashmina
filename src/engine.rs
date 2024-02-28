use std::{borrow::Cow, collections::HashMap};

use anyhow::Context;

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub kernels: HashMap<String, wgpu::ShaderModule>,
}

impl Engine {
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
    
        Ok(Self { device, queue, kernels })
    }
}
