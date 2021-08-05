#!/bin/bash

main() {
    log_i "Starting to generate synthetic data"
    python3 generate_data.py --init_round 1 --max_round 50
}


log_i() {
    log "[INFO][$(date '+%Y-%m-%d %H:%M:%S')] ${@}"
}


control_c() {
    echo ""
    exit
}

trap control_c SIGINT SIGTERM SIGHUP

main

exit
