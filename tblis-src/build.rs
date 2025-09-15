use std::path::PathBuf;

fn build_tblis() {
    // read environment variables
    // - TBLIS_SRC: source of tblis (should be git repository URL or path to local source)
    // - TBLIS_VER: version of tblis (`develop` for example)

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let default_src = format!("{manifest_dir}/external_deps/tblis");
    let tblis_src = std::env::var("TBLIS_SRC").unwrap_or(default_src);
    let tblis_ver = std::env::var("TBLIS_VER").unwrap_or("default".into());

    // TBLIS build both static and shared by default
    if cfg!(feature = "build_from_source") {
        let dst =
            cmake::Config::new("external_deps").define("TBLIS_SRC", tblis_src).define("TBLIS_VER", tblis_ver).build();
        // CMAKE_INSTALL_LIBDIR can be lib64 on some platforms
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    }
}

/// Generate link search paths from a list of paths.
///
/// This allows paths like `/path/to/lib1:/path/to/lib2` to be split into
/// individual paths.
fn generate_link_search_paths(paths: &str) -> Vec<String> {
    let split_char = if cfg!(windows) { ";" } else { ":" };
    paths.split(split_char).map(|path| path.to_string()).collect()
}

/// Generate root candidates for library search paths.
///
/// Code modified from
///
/// https://github.com/coreylowman/cudarc/blob/main/build.rs
fn root_candidates(env_candidates: &[&str]) -> Vec<PathBuf> {
    let root_candidates = ["/usr", "/usr/local", "/usr/local/share", "/opt"];

    env_candidates
        .iter()
        .map(|p| p.to_string())
        .map(std::env::var)
        .filter_map(Result::ok)
        .flat_map(|path| generate_link_search_paths(&path))
        .filter(|path| !path.is_empty())
        .chain(root_candidates.into_iter().map(|p| p.to_string()))
        .map(|p| p.into())
        .collect()
}

/// Generate candidates for library search paths.
///
/// Code modified from
///
/// https://github.com/coreylowman/cudarc/blob/main/build.rs
fn lib_candidates() -> impl Iterator<Item = PathBuf> {
    let lib_candidates = [
        "",
        "lib",
        "lib/stubs",
        "lib/x64",
        "lib/Win32",
        "lib/x86_64",
        "lib/x86_64-linux-gnu",
        "lib64",
        "lib64/stubs",
        "targets/x86_64-linux",
        "targets/x86_64-linux/lib",
        "targets/x86_64-linux/lib/stubs",
    ];
    lib_candidates.into_iter().map(|p| p.into())
}

fn path_candidates(env_candidates: &[&str]) -> impl Iterator<Item = PathBuf> {
    root_candidates(env_candidates)
        .into_iter()
        .flat_map(|root| lib_candidates().map(move |lib| root.join(lib)))
        .filter(|path| path.exists())
        .map(|path| std::fs::canonicalize(path).unwrap())
}

fn link_tblis() {
    let env_candidates = ["TBLIS_DIR", "REST_EXT_DIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"];
    // minimal rerun-if-env-changed to avoid unnecessary rebuilds
    println!("cargo:rerun-if-env-changed=TBLIS_DIR");
    for path in path_candidates(&env_candidates) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    if cfg!(feature = "static") {
        println!("cargo:rustc-link-lib=static=tblis");
    } else {
        println!("cargo:rustc-link-lib=tblis");
    }
}

fn main() {
    build_tblis();
    link_tblis();
}
