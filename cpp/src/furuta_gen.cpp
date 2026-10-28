// furuta_gen: generators for the Furuta pendulum (see esbo/commands.hpp for the subcommands).
#include "esbo/commands.hpp"
#include "furuta.hpp"

int main(int argc, char** argv) {
    return esbo::guarded_main(argc, argv, "furuta_gen", [](const std::string& cmd, const esbo::Args& a) {
        if (a.has("problem") && a.get("problem") != "furuta")
            throw std::runtime_error("furuta_gen only serves --problem furuta");
        return esbo::dispatch<esbo::furuta::Problem>(cmd, a, "furuta_gen");
    });
}
