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
          // xformers: true,   // uncomment this line if your project requires xformers
          triton: true,
          // sageattention: true   // uncomment this line if your project requires sageattention
        }
      }
    },
  ]
}
