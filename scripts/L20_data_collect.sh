#!/bin/bash
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..
export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_20
export AUTOPILOT_ROOT=${WORKSPACE}/carla_autopilot
export LEADERBOARD_ROOT=${WORKSPACE}/leaderboard_20/leaderboard
export SCENARIO_RUNNER_ROOT=${AUTOPILOT_ROOT}/leaderboard_20/scenario_runner_custom

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}
export PYTHONPATH=$PYTHONPATH:${AUTOPILOT_ROOT}

# general parameters
export PORT=2000
export TM_PORT=2500
export DEBUG_CHALLENGE=0

# simulation setup
dataset_name=training
export ROUTES=${WORKSPACE}/leaderboard_20/leaderboard/data/routes_${dataset_name}.xml
export ROUTES_SUBSET=0
export REPETITIONS=1

export CHALLENGE_TRACK_CODENAME=MAP
export TEAM_AGENT=${WORKSPACE}/lav/data_collect_L20/data_agent.py
export TEAM_CONFIG=${WORKSPACE}/lav/data_collect_L20/config.yaml
export TIME_STAMP=$(date +"%s")
export CHECKPOINT=${WORKSPACE}/logs/L20_training/route_${ROUTES_SUBSET}/log_${TIME_STAMP}.json
export PYTHON_FILE=${WORKSPACE}/lav/leaderboard_custom/leaderboard_evaluator.py


export RESUME=0
export TM_SEED=0

python ${PYTHON_FILE} \
    --port=${PORT} \
    --traffic-manager-port=${TM_PORT} \
    --routes=${ROUTES} \
    --routes-subset=${ROUTES_SUBSET} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME} \
    --traffic-manager-seed=${TM_SEED}
