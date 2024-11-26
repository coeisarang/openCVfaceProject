#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

int main() {
    // 얼굴 인식
    cv::CascadeClassifier face_cascade;
    face_cascade.load("c:/j/haarcascade_frontalface_default.xml");

    // 표정 인식을 위한 눈, 코, 입등의 위치 반환
    dlib::shape_predictor predictor;
    dlib::deserialize("c:/j/shape_predictor_68_face_landmarks.dat") >> predictor;

    // 표정 라벨링
    std::vector<std::string> expression_labels = { "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral" };

    // 표정 가중치 모델
    cv::Ptr<cv::ml::ANN_MLP> model = cv::ml::ANN_MLP::load("c:/j/emotion_model.xml");

    // 비디오 실행
    cv::VideoCapture video_capture(0);

    std::vector<cv::Rect> prev_faces;

    while (true) {
        // ret, frame 반환
        cv::Mat frame;
        video_capture >> frame;

        if (frame.empty()) {
            break;
        }

        // 얼굴인식을 위해 gray 변환
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 얼굴 인식
        // scaleFactor이 1에 가까울수록 표정 인식이 잘 되고 멀 수록 잘 안됨
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

        //region 얼굴이 인식되면 표정을 인식
        for (const auto& face : faces) {
            // 얼굴 크기에 알맞도록 사각형 그리기
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);

            // 얼굴 크기 반환
            cv::Mat face_roi = gray(face);

            // 표정을 인식하기 위해 표정 dataset과 똑같은 사이즈 변환
            // dataset 이미지와 입력된 얼굴의 크기가 다르면 error 발생
            cv::resize(face_roi, face_roi, cv::Size(64, 64));
            face_roi = face_roi / 255.0;

            // 모델을 통해 표정 분석
            cv::Mat input;
            face_roi.reshape(1, 1).convertTo(input, CV_32FC1);
            cv::Mat output;
            model->predict(input, output);

            // 해당 표정의 값 반환
            int expression_index = std::distance(output.begin<float>(), std::max_element(output.begin<float>(), output.end<float>()));

            // 표정에 따른 label 값 저장
            std::string expression_label = expression_labels[expression_index];
            // 표정 값 출력
            cv::putText(frame, expression_label, cv::Point(face.x, face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        }
        //endregion

        // 출력
        cv::imshow("Expression Recognition", frame);

        // esc 누를 경우 종료
        int key = cv::waitKey(25);
        if (key == 27) {
            break;
        }
    }

    if (video_capture.isOpened()) {
        video_capture.release();
    }
    cv::destroyAllWindows();

    return 0;
}