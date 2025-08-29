import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.util.Scanner;

public class YuzAlgilama {

    private static Net faceNet;
    private static Net ageNet;
    private static Net genderNet;

    private static final String[] GENDER_LIST = {"Erkek", "Kadin"};
    private static final String[] AGE_LIST = {"(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"};

    public static void main(String[] args) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("OpenCV yerel kütüphanesi yüklenemedi. Lütfen 'java.library.path' ayarını kontrol edin.");
            return;
        }

        if (!loadModels()) {
            System.err.println("Gerekli modeller yüklenemedi. Program sonlandırılıyor.");
            releaseResources();
            return;
        }

        Scanner scanner = new Scanner(System.in);
        String choice;

        while (true) {
            System.out.println("\n--- Yüz Algılama Menüsü ---");
            System.out.println("C: Kamera ile canlı algılama");
            System.out.println("V: Video dosyası ile algılama");
            System.out.println("Q: Programdan çıkış");
            System.out.print("Seçiminiz: ");
            choice = scanner.nextLine().trim().toLowerCase();

            switch (choice) {
                case "c":
                    System.out.println("Kamera modu başlatıldı. Çıkmak için pencereye tıklayıp ESC tuşuna basın.");
                    runDetection(new VideoCapture(0));
                    break;
                case "v":
                    System.out.println("Video modu başlatıldı.");
                    System.out.print("Lütfen video dosyasının yolunu girin: ");
                    String videoPath = scanner.nextLine();
                    runDetection(new VideoCapture(videoPath));
                    break;
                case "q":
                    System.out.println("Programdan çıkılıyor.");
                    releaseResources();
                    scanner.close();
                    return;
                default:
                    System.out.println("Geçersiz seçim. Lütfen 'C', 'V' veya 'Q' girin.");
                    break;
            }
        }
    }

    private static boolean loadModels() {
        System.out.println("Modeller yükleniyor...");
        try {
            faceNet = Dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
            ageNet = Dnn.readNetFromCaffe("age_net.prototxt", "age_net.caffemodel");
            genderNet = Dnn.readNetFromCaffe("gender_net.prototxt", "gender_net.caffemodel");

            if (faceNet.empty() || ageNet.empty() || genderNet.empty()) {
                System.err.println("Model dosyalarından biri boş. Dosya adlarını ve yollarını kontrol edin.");
                return false;
            }
            System.out.println("Tüm modeller başarıyla yüklendi.");
            return true;
        } catch (Exception e) {
            System.err.println("Modeller yüklenirken bir hata oluştu: " + e.getMessage());
            return false;
        }
    }

    private static void releaseResources() {
        faceNet = null;
        ageNet = null;
        genderNet = null;
    }

    private static void runDetection(VideoCapture source) {
        if (!source.isOpened()) {
            System.err.println("Akış kaynağı açılamadı! Lütfen bağlantıyı veya dosya yolunu kontrol edin.");
            return;
        }

        Mat frame = new Mat();
        String windowName = (source.get(0) == 0) ? "Kamera Yuz Algilama" : "Video Yuz Algilama";

        while (true) {
            source.read(frame);
            if (frame.empty()) {
                System.out.println("Akış bitti veya kare boş.");
                break;
            }

            Mat processedFrame = new Mat();
            Imgproc.resize(frame, processedFrame, new Size(300, 300));
            Mat blob = Dnn.blobFromImage(processedFrame, 1.0, new Size(300, 300), new Scalar(104, 177, 123), false, false);
            faceNet.setInput(blob);
            Mat detections = faceNet.forward();

            if (detections.dims() > 2) {
                detections = detections.reshape(1, detections.size(2));
            }

            int frameWidth = frame.cols();
            int frameHeight = frame.rows();

            for (int i = 0; i < detections.rows(); i++) {
                double confidence = detections.get(i, 2)[0];

                if (confidence > 0.5) {
                    int x1 = (int)(detections.get(i, 3)[0] * frameWidth);
                    int y1 = (int)(detections.get(i, 4)[0] * frameHeight);
                    int x2 = (int)(detections.get(i, 5)[0] * frameWidth);
                    int y2 = (int)(detections.get(i, 6)[0] * frameHeight);

                    x1 = Math.max(0, x1);
                    y1 = Math.max(0, y1);
                    x2 = Math.min(frameWidth - 1, x2);
                    y2 = Math.min(frameHeight - 1, y2);

                    if (x2 > x1 && y2 > y1) {
                        Rect faceBox = new Rect(x1, y1, x2 - x1, y2 - y1);
                        Mat faceROI = new Mat(frame, faceBox);

                        String gender = predictGender(faceROI);
                        String age = predictAge(faceROI);

                        String infoText = String.format("Cinsiyet: %s, Yas: %s", gender, age);

                        Imgproc.rectangle(frame, faceBox, new Scalar(0, 255, 0), 2);
                        Imgproc.putText(frame, infoText, new Point(x1, y1 - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 255, 0), 2);
                    }
                }
            }
            HighGui.imshow(windowName, frame);

            if (HighGui.waitKey(1) == 27) {
                break;
            }
        }

        source.release();
        HighGui.destroyAllWindows();
    }

    private static int getPredictionIndex(Mat face, Net net, Size size) {
        Mat blob = Dnn.blobFromImage(face, 1.0, size, new Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
        net.setInput(blob);
        Mat result = net.forward();
        Core.MinMaxLocResult mm = Core.minMaxLoc(result);
        return (int) mm.maxLoc.x;
    }

    private static String predictGender(Mat face) {
        int index = getPredictionIndex(face, genderNet, new Size(227, 227));
        return GENDER_LIST[index];
    }

    private static String predictAge(Mat face) {
        int index = getPredictionIndex(face, ageNet, new Size(227, 227));
        return AGE_LIST[index];
    }
}