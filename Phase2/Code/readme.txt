project_utils.py Task 1:
    python3 task1.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --model LBP --lsa_model PCA --k_features 20

Task 2:
    python3 task2.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --model LBP --lsa_model PCA --k_features 20  --m 20 --query_image Hand_0000589.jpg

Task 3:
    python3 task3.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --model LBP --lsa_model PCA --k_features 20  --label "without accessories"

Task 4:
    python3 task3.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --model LBP --lsa_model PCA --k_features 20  --label "without accessories" --m 20 --query_image  Hand_0000589.jpg

Task 5:
     python3 task5.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --model SIFT --lsa_model SVD --k_features 20 --label "dorsal" --query_image_path ~/ASU/MWDB/Project/DataSet/SmallerDataset/Hand_0008129.jpg

Task 6:
    python3 task6.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --model HOG --lsa_model SVD --subjectId 0

Task 7:
    python3 task7.py --image_folder ~/ASU/MWDB/Project/DataSet/testset1 --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --k 4

Task 8:
    python3 task8.py --metadata_file ~/ASU/MWDB/Project/DataSet/testset1_metadata.csv --k 4

Task extra_cred:
    python3 extra_cred.py --image_folder /home/dhruv/Allprojects/MWDB/Hand_small --metadata_file /home/dhruv/Allprojects/MWDB/HandInfo.csv --model CM --lsa_model PCA --k_features 20 --top_latent 5 --top_img 5 --extra_cred True
