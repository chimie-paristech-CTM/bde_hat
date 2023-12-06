for file in molecule*
do  
    xtb $file --ohess --cycles 1000 --chrg 0 -P 16 --uhf 0 > ${file%.xyz}.log
    cp xtbopt.xyz ${file%.xyz}_opt.xyz
    rm xtbrestart charges wbo xtbtopo.mol xtbopt.xyz xtbopt.log vibspectrum hessian xtbhess.xyz
done

