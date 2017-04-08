make:
	g++ main.cpp -o main -std=c++11 -I/usr/local/include/ -L/opt/vc/lib -lraspicam -lraspicam_cv -lmmal -lmmal_core -lmmal_util -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc -lpthread
test :
	g++ test.cpp -o test -std=c++11 -I/usr/local/include/ -L/opt/vc/lib -lraspicam -lraspicam_cv -lmmal -lmmal_core -lmmal_util -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_imgproc -lpthread
clean:
	rm main test