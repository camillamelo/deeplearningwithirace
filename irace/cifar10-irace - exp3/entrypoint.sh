#!/bin/bash

if [[ "$RUN_DEFAULT_EXP" = [tT][rR][uU][eE] ]]; then
    python main.py --dummy --clear-plots-models-and-datasets \
    echo -e "2018\n\n" | tee -a log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2018 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup | tee -a log.txt; \
    echo -e "\n\n\n\n2015\n\n" | tee -a log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --start-date 01/01/2015 --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup | tee -a log.txt; \
    echo -e "\n\n\n\nALL\n\n" | tee -a log.txt; \
    python main.py --train --eval --plot --plot-eval --save-plots --enrich-dataset --use-all-hyper-on-all-stocks --analyze-metrics --move-models-to-backup | tee -a log.txt \
    python main.py --dummy --restore-backups | tee -a log.txt; \
    echo -e "\n\n\nDONE\n" | tee -a log.txt
elif [[ "$RUN_IRACE_NAS" = [tT][rR][uU][eE] ]]; then
    python main.py --dummy --clear-plots-models-and-datasets
    (cd /code/irace ; /usr/local/lib/R/site-library/irace/bin/irace | tee ../log.txt)
    echo -e "\n\n\nDONE\n" | tee -a log.txt
elif [[ "$RUN_GENETIC_NAS" = [tT][rR][uU][eE] ]]; then
    python main.py --dummy --clear-plots-models-and-datasets
    python nas_genetic.py | tee log.txt
    echo -e "\n\n\nDONE\n" | tee -a log.txt
elif [[ "$RUN_PYMOO_NAS" = [tT][rR][uU][eE] ]]; then
    export NAS_STOCK="IBM"
    python main.py --dummy --clear-plots-models-and-datasets
    python nas_pymoo.py | tee log.txt
    echo -e "\n\n\nDONE IBM\n" | tee -a log.txt
    if [[ "$COMPRESS_AFTER_EXP" = [tT][rR][uU][eE] ]]; then
        tar -zcvf /code/exp_ibm.tar.gz /code/datasets /code/saved_models /code/saved_plots /code/irace /code/log.txt
    fi

    export NAS_STOCK="AAPL"
    python main.py --dummy --clear-plots-models-and-datasets
    python nas_pymoo.py | tee log.txt
    echo -e "\n\n\nDONE AAPL\n" | tee -a log.txt
    if [[ "$COMPRESS_AFTER_EXP" = [tT][rR][uU][eE] ]]; then
        tar -zcvf /code/exp_aapl.tar.gz /code/datasets /code/saved_models /code/saved_plots /code/irace /code/log.txt
    fi

    export NAS_STOCK="%5EBVSP"
    python main.py --dummy --clear-plots-models-and-datasets
    python nas_pymoo.py | tee log.txt
    echo -e "\n\n\nDONE BVSP\n" | tee -a log.txt
    if [[ "$COMPRESS_AFTER_EXP" = [tT][rR][uU][eE] ]]; then
        tar -zcvf /code/exp_bvsp.tar.gz /code/datasets /code/saved_models /code/saved_plots /code/irace /code/log.txt
    fi

    echo -e "\n\n\nDONE ALL\n"
    COMPRESS_AFTER_EXP="False"
fi

if [[ "$COMPRESS_AFTER_EXP" = [tT][rR][uU][eE] ]]; then
    tar -zcvf /code/exp.tar.gz /code/datasets /code/saved_models /code/saved_plots /code/irace /code/log.txt
fi

#tail -f /dev/null # to keep running
