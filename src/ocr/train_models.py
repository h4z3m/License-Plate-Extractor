from ocr.ocr import OCR

if __name__ == "__main__":
    classifier = OCR()
    classifier.load_dataset("../../data/Characters")

    print("Training RF classifier")
    classifier.train(mode="rf")
    classifier.save_trained_model("trained_model_rf.pk1")

    print("Training SVM classifier")
    classifier.train(mode="svm")
    classifier.save_trained_model("trained_model_svm.pk1")

    print("Training KNN classifier")
    classifier.train(mode="knn")
    classifier.save_trained_model("trained_model_knn.pk1")
