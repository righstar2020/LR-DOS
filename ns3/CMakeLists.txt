build_example(
  NAME lowrate-dos-simulation
  SOURCE_FILES lowrate-dos-simulation.cc lowrate-dos-application.cc legitimate-tcp-application.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libapplications}
    ${libcsma}
)
