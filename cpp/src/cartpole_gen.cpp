// cartpole_gen: generators for Cartpole (see esbo/commands.hpp for the subcommands).
#include "cartpole.hpp"
#include "esbo/commands.hpp"

int main(int argc, char** argv) {
    return esbo::guarded_main(argc, argv, "cartpole_gen", [](const std::string& cmd, const esbo::Args& a) {
        if (a.has("problem") && a.get("problem") != "cartpole")
            throw std::runtime_error("cartpole_gen only serves --problem cartpole");
        if (a.has("no_dead_time")) throw std::runtime_error("--no-dead-time exists for furuta only");
        return esbo::dispatch<esbo::cartpole::Problem>(cmd, a, "cartpole_gen");
    });
}
