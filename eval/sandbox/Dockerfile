# HumanEval sandbox for the Q200v2 kit: Python plus the kit's own sandbox driver at the path the runner execs.
# Built by eval/build_sandbox.sh from the fetched kit (the driver's sha256 is bound into every receipt).
FROM python:3.12-slim
COPY q200_sandbox_driver.py /opt/r0b0tlab/q200_sandbox_driver.py
RUN chmod 644 /opt/r0b0tlab/q200_sandbox_driver.py
ENTRYPOINT []
