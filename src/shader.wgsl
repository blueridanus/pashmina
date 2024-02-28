@group(0)
@binding(0)
var<storage, read_write> out: array<vec3u>;

@compute
@workgroup_size(8,8,8)
fn main(
    @builtin(local_invocation_id) local_pos: vec3<u32>,
    @builtin(global_invocation_id) global_pos: vec3<u32>,
){
    out[64 * global_pos.z + 8 * global_pos.y + global_pos.x] = global_pos;
}