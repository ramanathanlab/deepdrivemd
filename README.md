# DeepDriveMD
DeepDriveMD implemented with colmena

## Development
Installation:
```
make install
```

To run dev tools (isort, flake8, black, mypy):
```bash
make
make mypy
```

## Testing the workflow

The workflow can be tested locally using mock API's for the tasks by running:
```
python -m deepdrivemd.workflows.openmm_cvae --test -c tests/basic-local/test.yaml
```
This will generate an output directory for the run with logs, results, and task specific output folders.

Each test will write a timestamped run directory to the `runs/` directory specified in `tests/basic-local/test.yaml`.

To clean up the runs (by default these are ignored by git):
```
rm -r runs/
```
**Note**: Running the workflow requires that a Redis server is running.
After following the first step of the Installation Notes, you can start a redis
server which logs to `redis.log` by running:
```
nohup redis-server redis.conf --port 6379 &> redis.log &
```
To stop the server, run `kill <pid>` given the pid number from running `cat redis.log | grep pid`

**Note**: Mock testing is specified in each of the application scripts `deepdrivemd/applications/*/app.py`.
