fn main() {
    // Deployment may inject provenance, but local builds need no Git checkout
    // and never execute Git at build or runtime. Track changes for option_env!.
    println!("cargo:rerun-if-env-changed=BILLIARDS_BUILD_GIT_REVISION");
    println!("cargo:rerun-if-env-changed=BILLIARDS_BUILD_DIRTY");
}
