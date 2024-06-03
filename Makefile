build:
	zig build

.PHONY: release
release: build
	mkdir -p release
	cd zig-out/bin; tar -czf ../../release/ilu.tar.gz *
