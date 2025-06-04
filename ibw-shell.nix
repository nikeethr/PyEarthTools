{
  pkgs ? import <nixpkgs> { },
}:
pkgs.stdenv.mkDerivation rec {
  pname = "perftest";
  version = "4.5-0.20";
  src = pkgs.fetchFromGitHub {
    owner = "linux-rdma";
    repo = "perftest";
    rev = "v${version}";
    hash = "sha256-xLQNRCpOKW54+w4KzChtx8dxAKKuLOMvlwTP8Gf9N8M=";
  };

  nativeBuildInputs = [
    pkgs.libtool
    pkgs.autoconf
    pkgs.automake
  ];

  buildInputs = [
    pkgs.rdma-core
    pkgs.pciutils
  ];

  configurePhase = ''
    mkdir -p $TMP/install
    ./autogen.sh
    ./configure --prefix="$TMP/install"
  '';

  installPhase = ''
    make install
    chmod a+rX -R $TMP/install
    ls $TMP/install
    mkdir -p $out/bin
    cp -r $TMP/install/* $out/
  '';
}
