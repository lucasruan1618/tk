module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '/workspace/miniconda3/envs/404-base-miner-env/bin/python',
    args: '--port 10006'
  }]
};
