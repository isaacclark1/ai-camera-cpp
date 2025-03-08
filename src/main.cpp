#include "server/Server.hpp"

int main(int argc, char** argv) {
  Server &server = Server::getInstance();
  server.startServer(9898);
  return 0;
}