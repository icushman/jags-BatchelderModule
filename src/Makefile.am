SUBDIRS = functions

jagsmod_LTLIBRARIES = batchelder.la

batchelder_la_SOURCES = BatchelderModule.cc

batchelder_la_CPPFLAGS = -I$(includedir)/JAGS

batchelder_la_LIBADD = functions/batchelderfunc.la
if WINDOWS
batchelder_la_LIBADD += -ljags-$(JAGS_MAJOR)
else
batchelder_la_LIBADD += -ljags
endif

batchelder_la_LDFLAGS = -module -avoid-version
if WINDOWS
batchelder_la_LDFLAGS += -no-undefined
endif
