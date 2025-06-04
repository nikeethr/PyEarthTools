{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      # NOTE: this will cause a rebuild of everything if the right gcc version isn't matched.
      # replaceStdenv = ({ pkgs }: pkgs.gcc11Stdenv);
    };
  },
}:
let

  inherit (pkgs.cudaPackages)
    backendStdenv
    cuda_cccl
    cuda_cudart
    cuda_nvcc
    cuda_nvml_dev
    cudaAtLeast
    cudaOlder
    cudatoolkit
    flags
    ;

  inherit (pkgs.lib) getLib;
in
backendStdenv.mkDerivation rec {
  pname = "nvbandwidth";
  version = "0.8";
  src = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "nvbandwidth";
    rev = "v${version}";
    hash = "sha256-PhJY7F0aGNoejLlhSNT3p3PjYKfywCq2nZGvHTu0Q/8=";
  };

  nativeBuildInputs =
    [
      pkgs.autoAddDriverRunpath
      pkgs.cmake
      pkgs.python3
      pkgs.which
      pkgs.makeWrapper
    ]
    ++ pkgs.lib.optionals (cudaOlder "11.4") [ cudatoolkit ]
    ++ pkgs.lib.optionals (cudaAtLeast "11.4") [ cuda_nvcc ];

  buildInputs =
    pkgs.lib.optionals (cudaOlder "11.4") [ cudatoolkit ]
    ++ pkgs.lib.optionals (cudaAtLeast "11.4") [
      cuda_cudart
      cuda_nvcc # crt/host_config.h
      cuda_nvml_dev
    ]
    ++ [
      pkgs.boost
      pkgs.linuxPackages.nvidia_x11
    ]
    # NOTE: CUDA versions in Nixpkgs only use a major and minor version. When we do comparisons
    # against other version, like below, it's important that we use the same format. Otherwise,
    # we'll get incorrect results.
    # For example, lib.versionAtLeast "12.0" "12.0.0" == false.
    ++ pkgs.lib.optionals (cudaAtLeast "12.0") [ cuda_cccl ];

  postPatch = ''
    substituteInPlace CMakeLists.txt \
         --replace-fail 'CMAKE_SYSTEM_NAME STREQUAL "Linux"' \
                        'CMAKE_SYSTEM_NAME STREQUAL "Potato"' \
         --replace-fail 'set(Boost_USE_STATIC_LIBS ON)' \
                        'set(Boost_USE_STATIC_LIBS OFF)' \
  '';

  # will be required for multinode - see docs
  # -DMULTINODE=1
  cmakeFlags = [
    "-DBoost_USE_STATIC_LIBS=OFF"
    "-DCMAKE_BUILD_TYPE=Debug"
  ];
  dontStrip = true;

  installPhase = ''
    mkdir -p $out/bin
    cp ./nvbandwidth $out/bin
    chmod a+rx $out/bin/nvbandwidth
    wrapProgram "$out/bin/nvbandwidth" \
      --prefix LD_LIBRARY_PATH ":" "${getLib pkgs.linuxPackages.nvidia_x11}"
  '';

  separateDebugInfo = true;

  # installPhase - this needs to be done manually...
  # quicker alternative to passthru
  # nativeInstallCheckInputs = [ pkgs.versionCheckHook ];
  # doInstallCheck = true;
}
