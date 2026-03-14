module.exports = {
  run: [

    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },

    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
          flashattention: true,
          triton: true,
        }
      }
    },

    // audio-separator: GPU for nvidia, CPU for everything else
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install audio-separator[gpu]"
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install audio-separator[cpu]"
      }
    },

  ]
}
