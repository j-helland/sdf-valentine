// @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
struct VertexOut {
    @builtin(position) position_clip: vec4<f32>,
    @location(0) coord: vec2<f32>,
}

@vertex
fn main(
    @builtin(vertex_index) VertexIndex: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
) -> VertexOut {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );

    var output: VertexOut;
    output.position_clip = vec4(pos[VertexIndex], 0.0, 1.0);
    // output.position_clip = vec4(position, 1.0) * object_to_clip;
    output.coord = vec2(position.x, position.y);
    return output;
}