#include <module/Module.h> // include JAGS module base class
#include <functions/Batchelder.h> // include Take The Best function class



namespace jags {
namespace batchelder { // start defining the module namespace

  // Module class
  class BatchelderModule : public Module {
    public:
      BatchelderModule(); // constructor
      ~BatchelderModule(); // destructor
  };

  // Constructor function
  BatchelderModule::BatchelderModule() : Module("batchelder") {
    insert(new BATCHELDER);

  }

  // Destructor function
  BatchelderModule::~BatchelderModule() {
    std::vector<Distribution*> const &dvec = distributions();
    for (unsigned int i = 0; i < dvec.size(); ++i) {
      delete dvec[i]; // delete all instantiated distribution objects
    }

    std::vector<Function*> const &fvec = functions();
    for (unsigned int i = 0; i < fvec.size(); ++i) {
      delete fvec[i];
    }
  }

} // end namespace definition
}

jags::batchelder::BatchelderModule _batchelder_module;
