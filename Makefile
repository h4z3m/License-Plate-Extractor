run:
	python src/main.py\
		-lc ./src/config/logging_config.json\
		-pc ./src/config/plate_extraction_config.json\
		-ec ./src/config/lpe_config.json\
		-dp ./data/Vehicles\
		-n 100\
		-o ./data/output
