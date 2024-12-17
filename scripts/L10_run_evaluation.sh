#!/bin/bash
export SHELL_PATH=$(dirname $(readlink -f $0))
export WORKSPACE=${SHELL_PATH}/..
export CARLA_ROOT=${WORKSPACE}/CARLA_Leaderboard_10
export LEADERBOARD_ROOT=${WORKSPACE}/leaderboard_10/leaderboard
export SCENARIO_RUNNER_ROOT=${WORKSPACE}/leaderboard_10/scenario_runner

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SCENARIO_RUNNER_ROOT}

# general parameters
export PORT=2000
export TM_PORT=2500
export SEED=42
# export SEED=11037
# export SEED=114514
export DEBUG_CHALLENGE=0

export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes_testing.xml
export ROUTES_SUBSET=0
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export TIME_STAMP=$(date +"%s")
export CHECKPOINT_ENDPOINT=${WORKSPACE}/logs/L10_testing/route_${ROUTES_SUBSET}_seed_${SEED}_${TIME_STAMP}.json
export TEAM_AGENT=${WORKSPACE}/team_code/lav_agent.py
export TEAM_CONFIG=${WORKSPACE}/team_code/config.yaml

RESUME=0

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
    --port=${PORT} \
    --trafficManagerPort=${TM_PORT} \
    --trafficManagerSeed=${SEED} \
    --scenarios=${SCENARIOS}  \
    --routes=${ROUTES} \
    --routes-subset=${ROUTES_SUBSET} \
    --repetitions=${REPETITIONS} \
    --track=${CHALLENGE_TRACK_CODENAME} \
    --checkpoint=${CHECKPOINT_ENDPOINT} \
    --agent=${TEAM_AGENT} \
    --agent-config=${TEAM_CONFIG} \
    --debug=${DEBUG_CHALLENGE} \
    --resume=${RESUME}
