{
  description = "MLflow Dev Shell with Python and NumPy support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default =
          (pkgs.buildFHSEnv {
            name = "mlflow-shell";

            targetPkgs = pkgs:
              with pkgs; [
                python3
                uv
                zlib
                libffi
                stdenv.cc.cc.lib # for libstdc++
                git
              ];
          }).env;
      }
    );
}
