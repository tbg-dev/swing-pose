#include <iostream>

#include <boost/program_options.hpp>
#include <c4a/core/log.h>
#include <c4a/pose/IBodyFromImage.h>

static constexpr auto LGR = "main";

//-----------------------------------------------------------------------------
// Entry point when not running as a Windows background application.
//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Define program options
    boost::program_options::options_description desc("Options");
    desc.add_options()
            ("help,h"     ,                                                           "Print help messages")
            ("filename,f" , boost::program_options::value<std::string>()->required(), "Input filename"     );

    // Parse program options
    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc), vm
        );
        boost::program_options::notify(vm);

        // Print help if requested
        if (vm.count("help")) {
            std::cout << std::endl << desc;
            return EXIT_SUCCESS;
        }

    } catch (boost::program_options::error& e) {
        std::cerr << e.what();
        std::cout << std::endl << desc;
        return EXIT_FAILURE;
    }

    // Trace logging
    c4a::core::Logging::setLogLevel(c4a::core::Logging::Level::TRACE);

    // Run the algorithm
    auto bodyFromImage = c4a::pose::IBodyFromImage::create();
    bodyFromImage->run(vm["filename"].as<std::string>().c_str());

    return EXIT_SUCCESS;
}