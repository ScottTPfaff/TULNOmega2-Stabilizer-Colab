{
  description = "TUNL Omega 2 - Flask/CuPy Python app with reproducible Nix environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.pip
          ps.setuptools
          ps.flask
          ps.gunicorn
          ps.werkzeug
          ps.numpy
          ps.scipy
          ps.matplotlib
          # Use CUDA-enabled cupy; replace with your CUDA version if needed
          ps.cupy
          ps.autopep8
          ps.pip
          ps.flask
        ]);
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
          ];
          shellHook = ''
            echo "Python 3.12 environment with Flask, CuPy (CUDA 12.x), numpy, scipy, matplotlib, etc."
            echo "Activate with 'nix develop', then run 'python main.py'"
          '';
        };
      }
    );
}
