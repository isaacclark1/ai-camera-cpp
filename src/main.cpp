#include "global/log_colours.hpp"
#include "global/io_mutex.hpp"
#include "server/Server.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include "DetectionInference.hpp"
#include "Camera.hpp"
#include "server/Server.hpp"

static inline void printApplicationStartBanner(
  std::string app_name,
  std::string network_file,
  int frame_height,
  int frame_width,
  int jpeg_frame_quality,
  int frame_skip_factor
)
{
  using namespace std;
  cout << BOLDWHITE;
  cout << "====================================================================================" << "\n";
  cout << "                                                                                    " << "\n";
  cout << "     STARTING APPLICATION:  " << app_name                                             << "\n";
  cout << "                                                                                    " << "\n";
  cout << "====================================================================================" << "\n";
  cout << "     LOG COLOURS:                                                                   " << "\n" << RESET << BOLDGREEN;
  cout << "                   Camera                                                           " << "\n" << RESET << BOLDBLUE;
  cout << "                   DetectionInference                                               " << "\n" << RESET << BOLDMAGENTA;
  cout << "                   Server                                                           " << "\n" << RESET << BOLDYELLOW;
  cout << "                   Warnings                                                         " << "\n" << RESET << BOLDRED;
  cout << "                   Errors                                                           " << "\n" << RESET << BOLDWHITE;
  cout << "====================================================================================" << "\n";
  cout << "     NETWORK FILE: "       << network_file                                            << "\n";
  cout << "     FRAME HEIGHT: "       << frame_height                                            << "\n";
  cout << "     FRAME WIDTH: "        << frame_width                                             << "\n";
  cout << "     JPEG FRAME QUALITY: " << jpeg_frame_quality                                      << "\n";
  cout << "     FRAME SKIP FACTOR: "  << frame_skip_factor                                       << "\n";
  cout << "====================================================================================" << "\n" << endl << RESET;
}

constexpr uint16_t DEFAULT_FRAME_HEIGHT = 1080;
constexpr uint16_t DEFAULT_FRAME_WIDTH = 1920;
constexpr uint8_t DEFAULT_JPEG_FRAME_QUALITY = 50;
constexpr uint8_t DEFAULT_FRAME_SKIP_FACTOR = 1;

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cerr << BOLDRED << "Error: No network file provided" << "\n";
    std::cerr << "Usage: " << argv[0];
    std::cerr << " <network_file> [frame_height, frame_width, jpeg_frame_quality, frame_skip_factor]";
    std::cerr << std::endl << RESET;

    return 1;
  }
  
  std::string network_file = argv[1];
  uint16_t frame_height = DEFAULT_FRAME_HEIGHT;
  uint16_t frame_width = DEFAULT_FRAME_WIDTH;
  uint8_t jpeg_frame_quality = DEFAULT_JPEG_FRAME_QUALITY;
  uint8_t frame_skip_factor = DEFAULT_FRAME_SKIP_FACTOR;

  try {
    if (argc >= 3) frame_height = static_cast<uint16_t>(std::stoi(argv[2]));
    if (argc >= 4) frame_width = static_cast<uint16_t>(std::stoi(argv[3]));
    if (argc >= 5) jpeg_frame_quality = static_cast<uint8_t>(std::stoi(argv[4]));
    if (argc >= 6) frame_skip_factor = static_cast<uint8_t>(std::stoi(argv[5]));
  }
  catch (const std::exception &e) {
    std::cerr << BOLDRED << "Error: Invalid argument format. " << e.what() << std::endl << RESET;
    std::cerr << "Usage: " << argv[0];
    std::cerr << " <network_file> [frame_height, frame_width, jpeg_frame_quality, frame_skip_factor]";
    return 1;
  }

  printApplicationStartBanner(
    argv[0],
    network_file,
    static_cast<int>(frame_height),
    static_cast<int>(frame_width),
    static_cast<int>(jpeg_frame_quality),
    static_cast<int>(frame_skip_factor)
  );

  try {
    DetectionInference::getInstance(
      network_file, frame_height, frame_width
    );
    Camera::getInstance(
      frame_height, frame_width, frame_skip_factor
    );
    Server &server = Server::getInstance(jpeg_frame_quality);
  
    server.startServer(9898);
  }
  catch (const std::runtime_error &e) {
    std::cerr << BOLDRED << "❌ Runtime error caught in main(): " << e.what() << std::endl << RESET;
  }
  catch (const std::exception &e) {
    std::cerr << BOLDRED << "❌ Exception caught in main(): " << e.what() << std::endl << RESET;
  }
  catch (...) {
    std::cerr << BOLDRED << "❌ Unknown error caught in main()" << std::endl << RESET;
  }
  
  return 0;
}