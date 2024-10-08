from app.main.consumer.pipeline_consumer import PipelineConsumer
from app.main.util.register import init_processors

if __name__ == '__main__':
    for i in range(1, 10):
        config_filename = f'config_{i}.yaml'
        print(f"Running pipeline with {config_filename}")
        
        processors = init_processors(config_filename)
        pipeline_consumer = PipelineConsumer(processors, config_filename)
        pipeline_consumer.main()

# if __name__ == '__main__':
#     processors = init_processors()
#     pipeline_consumer = PipelineConsumer(processors)
#     pipeline_consumer.main()
