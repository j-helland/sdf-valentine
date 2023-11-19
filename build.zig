const builtin = @import("builtin");
const std = @import("std");

const zgpu = @import("libs/zgpu/build.zig");
const zmath = @import("libs/zmath/build.zig");
const zpool = @import("libs/zpool/build.zig");
const zglfw = @import("libs/zglfw/build.zig");
const zgui = @import("libs/zgui/build.zig");

const content_dir = "assets/";

var zmath_pkg: zmath.Package = undefined;
var zpool_pkg: zpool.Package = undefined;
var zglfw_pkg: zglfw.Package = undefined;
var zgui_pkg: zgui.Package = undefined;
var zgpu_pkg: zgpu.Package = undefined;

pub const Options = struct {
    optimize: std.builtin.Mode,
    target: std.zig.CrossTarget,
};

pub fn build(b: *std.Build) void {
    const options = Options{
        .optimize = b.standardOptimizeOption(.{}),
        .target = b.standardTargetOptions(.{}),
    };
    const exe = b.addExecutable(.{
        .name = "zsdf",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = options.target,
        .optimize = options.optimize,
    });

    const exe_options = b.addOptions();
    exe.addOptions("build_options", exe_options);
    exe_options.addOption([]const u8, "content_dir", content_dir);

    const install_content_step = b.addInstallDirectory(.{
        .source_dir = .{ .path = content_dir },
        .install_dir = .{ .custom = "" },
        .install_subdir = "bin/" ++ content_dir,
    });
    exe.step.dependOn(&install_content_step.step);

    // Linking
    packageCrossPlatform(b, options);
    zmath_pkg.link(exe);
    zgpu_pkg.link(exe);
    zgui_pkg.link(exe);
    zglfw_pkg.link(exe);

    // Install
    const install_step = b.step("install-zsdl", "Install");
    install_step.dependOn(&b.addInstallArtifact(exe, .{}).step);

    // Run
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(install_step);

    const run_step = b.step("run", "Run");
    run_step.dependOn(&run_cmd.step);

    b.getInstallStep().dependOn(install_step);
}

fn packageCrossPlatform(b: *std.Build, options: Options) void {
    const target = options.target;
    const optimize = options.optimize;

    zmath_pkg = zmath.package(b, target, optimize, .{});
    zpool_pkg = zpool.package(b, target, optimize, .{});
    zglfw_pkg = zglfw.package(b, target, optimize, .{});
    zgui_pkg = zgui.package(b, target, optimize, .{
        .options = .{ .backend = .glfw_wgpu },
    });
    zgpu_pkg = zgpu.package(b, target, optimize, .{
        .options = .{ .uniforms_buffer_size = 4 * 1024 * 1024 },
        .deps = .{ .zpool = zpool_pkg.zpool, .zglfw = zglfw_pkg.zglfw },
    });
}
