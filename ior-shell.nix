{ pkgs ? import <nixpkgs> {} }:
pkgs.ior.overrideAttrs (finalAttrs: previousAttrs: {
  pname = "ior-re";
  src = pkgs.fetchFromGitHub {
    owner = "hpc";
    repo = "ior";
    rev = finalAttrs.version;
    hash = "sha256-w6pxJIjl7LSgOUSJPr4uY+ZtgDweUKexSo8X6msqgpQ=";
  };
  version = "3.3.0rc1";
})
