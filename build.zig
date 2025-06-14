const builtin = @import("builtin");
const std = @import("std");

const content_dir = "assets/";

pub const Options = struct {
    optimize: std.builtin.Mode = .ReleaseSmall,
    target: std.Build.ResolvedTarget,
};

pub fn build(b: *std.Build) void {
    const options = Options{ .target = b.resolveTargetQuery(.{}) };
    const exe = b.addExecutable(.{
        .name = "ilu",
        .root_source_file = .{ .cwd_relative = "src/main.zig" },
        .target = options.target,
        .optimize = options.optimize,
    });

    const zglfw = b.dependency("zglfw", .{
        .target = options.target,
    });
    exe.root_module.addImport("zglfw", zglfw.module("root"));
    exe.linkLibrary(zglfw.artifact("glfw"));

    @import("zgpu").addLibraryPathsTo(exe);
    const zgpu = b.dependency("zgpu", .{
        .target = options.target,
    });
    exe.root_module.addImport("zgpu", zgpu.module("root"));
    exe.linkLibrary(zgpu.artifact("zdawn"));

    const zgui = b.dependency("zgui", .{
        .target = options.target,
        .backend = .glfw_wgpu,
    });
    exe.root_module.addImport("zgui", zgui.module("root"));
    exe.linkLibrary(zgui.artifact("imgui"));

    const zmath = b.dependency("zmath", .{
        .target = options.target,
    });
    exe.root_module.addImport("zmath", zmath.module("root"));

    const zpool = b.dependency("zpool", .{
        .target = options.target,
    });
    exe.root_module.addImport("zpool", zpool.module("root"));

    if (options.target.result.os.tag == .macos) {
        if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
            exe.addLibraryPath(system_sdk.path("macos12/usr/lib"));
            exe.addSystemFrameworkPath(system_sdk.path("macos12/System/Library/Frameworks"));
        }
    } else if (options.target.result.os.tag == .linux) {
        if (b.lazyDependency("system_sdk", .{})) |system_sdk| {
            exe.addLibraryPath(system_sdk.path("linux/lib/x86_64-linux-gnu"));
        }
    }

    //const install_content_step = b.addInstallDirectory(.{
    //    .source_dir = b.path(content_dir),
    //    .install_dir = .{ .custom = "" },
    //    .install_subdir = b.pathJoin(&.{ "bin", content_dir }),
    //});
    //exe.step.dependOn(&install_content_step.step);

    const exe_options = b.addOptions();
    exe.root_module.addOptions("build_options", exe_options);
    //exe_options.addOption([]const u8, "content_dir", content_dir);

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
