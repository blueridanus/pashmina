@group(0) @binding(0)
var<storage, read_write> buf: array<u32>;

@group(0) @binding(1)
var<storage, read_write> prev: array<u32>;

// must be the same as psum1.wgsl
const WG_LEN : u32 = 256;

@compute @workgroup_size(WG_LEN)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(workgroup_id) wg_id: vec3u,
){
    buf[global_id.x + WG_LEN] += prev[wg_id.x];
}