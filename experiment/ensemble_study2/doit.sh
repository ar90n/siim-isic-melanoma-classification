poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b0_mlp_1/cv_test_en_b0_mlp.csv
mv ./submission.csv ./submission_b0.csv

poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b1_mlp_1/cv_test_en_b1_mlp.csv
mv ./submission.csv ./submission_b1.csv

poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b2_mlp_2/cv_test_en_b2_mlp.csv
mv ./submission.csv ./submission_b2.csv

poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b3_mlp_1/cv_test_en_b3_mlp.csv
mv ./submission.csv ./submission_b3.csv

poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b4_mlp_1/cv_test_en_b4_mlp.csv
mv ./submission.csv ./submission_b4.csv

poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b5_mlp_1/cv_test_en_b5_mlp.csv
mv ./submission.csv ./submission_b5.csv

poetry run python ensemble2.py ../../../input/my-isic2020-experiments/en_b6_mlp_1/cv_test_en_b6_mlp.csv
mv ./submission.csv ./submission_b6.csv
