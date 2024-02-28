@group(0) @binding(0)
var<storage, read_write> buf: array<u32>;

@group(0) @binding(1)
var<storage, read_write> next: array<u32>;

const WG_LEN : u32 = 256;
var<workgroup> scratchpad: array<u32, WG_LEN>;

@compute @workgroup_size(WG_LEN)
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) wg_id: vec3u,
){
    var sum = buf[global_id.x];
    scratchpad[local_id.x] = sum;
    workgroupBarrier();
    
    for(var i = 0u; i < firstTrailingBit(WG_LEN); i += 1u){
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += scratchpad[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        scratchpad[local_id.x] = sum;
    }

    buf[global_id.x] = sum;

    if local_id.x == WG_LEN - 1 {
        next[wg_id.x] = sum;
    }
}