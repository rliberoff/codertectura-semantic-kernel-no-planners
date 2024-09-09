#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010
#pragma warning disable SKEXP0060

#pragma warning disable S3903 // Types should be defined in named namespaces

using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;

using Azure;
using Azure.AI.OpenAI;

using Encamina.Enmarcha.Core.DataAnnotations;

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Planning;
using Microsoft.SemanticKernel.Planning.Handlebars;
using Microsoft.SemanticKernel.TextToImage;

var builder = Host.CreateApplicationBuilder(args);

builder.Configuration.SetBasePath(Directory.GetCurrentDirectory());

/* Load Configuration */

if (Debugger.IsAttached)
{
    builder.Configuration.AddJsonFile(@"appsettings.debug.json", optional: true, reloadOnChange: true);
}

builder.Configuration.AddJsonFile($@"appsettings.{builder.Environment.EnvironmentName}.json", optional: true, reloadOnChange: true)
                     .AddJsonFile($@"appsettings.{Environment.UserName}.json", optional: true, reloadOnChange: true)
                     .AddEnvironmentVariables();

// Configure logging to filter out specific traces, allowing only `Console.Write` and `Console.WriteLine` outputs...
builder.Logging.ClearProviders()
               .AddConsole()
               .AddFilter(@"Microsoft.Hosting.Lifetime", LogLevel.None)
               .AddFilter(@"System.Net.Http.HttpClient.Default.ClientHandler", LogLevel.None)
               .AddFilter(@"System.Net.Http.HttpClient.Default.LogicalHandler", LogLevel.None)
               ;

builder.Services.AddOptionsWithValidateOnStart<AzureOpenAIOptions>().Bind(builder.Configuration.GetSection(nameof(AzureOpenAIOptions))).ValidateDataAnnotations();
builder.Services.AddOptionsWithValidateOnStart<WeatherStackOptions>().Bind(builder.Configuration.GetSection(nameof(WeatherStackOptions))).ValidateDataAnnotations();

builder.Services.AddHttpClient()
                .AddSingleton<TimePlugin>()
                .AddTransient<ImagePlugin>()
                .AddTransient<WeatherPlugin>()
                ;

builder.Services.AddTransient(serviceProvider =>
{
    var oaiOptions = serviceProvider.GetRequiredService<IOptions<AzureOpenAIOptions>>().Value;

    var oaiClient = new OpenAIClient(oaiOptions.Endpoint, new AzureKeyCredential(oaiOptions.Key), new OpenAIClientOptions(oaiOptions.ServiceVersion));

    var kernelBuilder = Kernel.CreateBuilder();

    kernelBuilder.AddAzureOpenAITextToImage(oaiOptions.ImageGenerationModelDeploymentName, oaiClient)
                 .AddAzureOpenAIChatCompletion(oaiOptions.ChatModelDeploymentName, oaiClient);

    var kernel = kernelBuilder.Build();

    return kernel;
});

using var cancellationTokenSource = new CancellationTokenSource();
using var host = builder.Build();

var kernel = host.Services.GetRequiredService<Kernel>();

var imagePlugin = host.Services.GetRequiredService<ImagePlugin>();
var timePlugin = host.Services.GetRequiredService<TimePlugin>();
var weatherPlugin = host.Services.GetRequiredService<WeatherPlugin>();

kernel.ImportPluginFromObject(imagePlugin);
kernel.ImportPluginFromObject(timePlugin);
kernel.ImportPluginFromObject(weatherPlugin);

const string Goal = @"Check current UTC time, then tell me the current weather in Madrid city, and finally use that information from the weather to create an image.";

await AppAsync(kernel, cancellationTokenSource.Token);

await host.RunAsync(cancellationTokenSource.Token);

//** APP **//
static async Task AppAsync(Kernel kernel, CancellationToken cancellationToken)
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.WriteLine(@"Starting App...");

    await UseFunctionCallingStepwisePlanner(kernel, cancellationToken);
    await UseHandlebarsPlanner(kernel, cancellationToken);
    await UseKernel(kernel, cancellationToken);

    Console.ForegroundColor = ConsoleColor.Red;
    Console.WriteLine("\nPress any key to close...");
    Console.ReadLine();
    Console.ResetColor();
    Console.WriteLine("Bye!");

    Environment.Exit(0);
}

//** ACTIONS **//

static async Task UseFunctionCallingStepwisePlanner(Kernel kernel, CancellationToken cancellationToken)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("\n—————————————————————————————————————————————");
    Console.WriteLine("Starting Function Calling Stepwise Planner...");

    var stopwatch = Stopwatch.StartNew();

    FunctionCallingStepwisePlanner planner = new();
    var plannerResult = await planner.ExecuteAsync(kernel, Goal, cancellationToken: cancellationToken);
    
    stopwatch.Stop();

    ShowFunctionCallingStepwisePlannerResults(plannerResult, stopwatch.Elapsed.Seconds);
}

static async Task UseHandlebarsPlanner(Kernel kernel, CancellationToken cancellationToken)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("\n—————————————————————————————————————————————");
    Console.WriteLine("Starting Handlebars Planner...");

    var stopwatch = Stopwatch.StartNew();

    HandlebarsPlanner planner = new();
    var plan = await planner.CreatePlanAsync(kernel, Goal, cancellationToken: cancellationToken);
    var result = await plan.InvokeAsync(kernel, cancellationToken: cancellationToken);

    stopwatch.Stop();

    ShowHandlebarsPlannerResults(plan, result, stopwatch.Elapsed.Seconds);
}

static async Task UseKernel(Kernel kernel, CancellationToken cancellationToken)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("\n—————————————————————————————————————————————");
    Console.WriteLine(@"Starting Kernel as planner...");

    var stopwatch = Stopwatch.StartNew();

    ChatHistory chatHistory = new();
    chatHistory.AddUserMessage(Goal);

    OpenAIPromptExecutionSettings executionSettings = new() { ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions };

    var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

    var result = await chatCompletionService.GetChatMessageContentAsync(chatHistory, executionSettings, kernel, cancellationToken);

    stopwatch.Stop();

    ShowKernelResults(result, chatHistory, stopwatch.Elapsed.Seconds);
}

static void ShowFunctionCallingStepwisePlannerResults(FunctionCallingStepwisePlannerResult result, int elapsedSeconds)
{
    ShowPlanFromChatHistory(result.ChatHistory!);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("\nPlanner execution result:");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine(result.FinalAnswer);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"\nPlanner execution total time: {elapsedSeconds} seconds");

    Console.ResetColor();
}

static void ShowHandlebarsPlannerResults(HandlebarsPlan plan, string result, int elapsedSeconds)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine(@"The plan is:");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine(plan);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("\nPlanner execution result:");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine(result);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"\nPlanner execution total time: {elapsedSeconds} seconds");

    Console.ResetColor();
}

static void ShowKernelResults(ChatMessageContent result, ChatHistory plan, int elapsedSeconds)
{
    ShowPlanFromChatHistory(plan);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("\nKernel execution result:");
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.WriteLine(result);

    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine($"\nKernel execution total time: {elapsedSeconds} seconds");

    Console.ResetColor();
}

static void ShowPlanFromChatHistory(ChatHistory chatHistory)
{
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine("The plan is:\n");
    foreach (var item in chatHistory)
    {
        ConsoleColor roleColor;

        if (item.Role == AuthorRole.System)
        {
            roleColor = ConsoleColor.DarkBlue;
        }
        else if (item.Role == AuthorRole.Tool)
        {
            roleColor = ConsoleColor.DarkMagenta;
        }
        else if (item.Role == AuthorRole.User)
        {
            roleColor = ConsoleColor.DarkGreen;
        }
        else if (item.Role == AuthorRole.Assistant)
        {
            roleColor = ConsoleColor.DarkYellow;
        }
        else
        {
            roleColor = ConsoleColor.DarkGray;
        }

        Console.BackgroundColor = roleColor;
        Console.ForegroundColor = ConsoleColor.White;
        Console.Write($"{item.Role.Label}");

        Console.ResetColor();
        Console.ForegroundColor = roleColor;
        Console.WriteLine($"\n{item.Content!}\n");

        Console.ResetColor();
    }
}

//** AI PLUGINS **//

internal sealed class TimePlugin
{
    [KernelFunction]
    [Description("Retrieves the current time in UTC")]
    public string GetCurrentUtcTime() => DateTime.UtcNow.ToString("R");
}

internal sealed class WeatherPlugin
{
    private readonly IChatCompletionService chatCompletionService;
    private readonly IHttpClientFactory httpClientFactory;
    private readonly WeatherStackOptions options;

    public WeatherPlugin(Kernel kernel, IHttpClientFactory httpClientFactory, IOptions<WeatherStackOptions> options)
    {
        this.chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
        this.httpClientFactory = httpClientFactory;

        this.options = options.Value;
    }

    [KernelFunction]
    [Description("Gets the current weather for the specified city")]
    public async Task<string> GetWeatherForCityAsync(string cityName, CancellationToken cancellationToken)
    {
        var client = httpClientFactory.CreateClient();

        using var request = new HttpRequestMessage(HttpMethod.Get, $@"https://api.weatherstack.com/current?query={cityName}&access_key={options.AccessKey}");
        using var response = await client.SendAsync(request, cancellationToken);
        response.EnsureSuccessStatusCode();

        var content = await response.Content.ReadAsStringAsync(cancellationToken);

        var systemPrompt = $$"""
            You are an expert AI understanding and interpreting the JSON response from the Weatherstack service. Create a easily to read and short summary of the weather from the following JSON:
            
            {{content}}
            
            """;

        var result = await chatCompletionService.GetChatMessageContentAsync(systemPrompt, new OpenAIPromptExecutionSettings()
        {
            MaxTokens = 200,
            Temperature = 0.1,
            TopP = 1.0,
        }, cancellationToken: cancellationToken);

        return result.Content!;
    }
}

internal sealed class ImagePlugin
{
    private readonly IChatCompletionService chatCompletionService;
    private readonly ITextToImageService textToImageService;

    public ImagePlugin(Kernel kernel)
    {
        this.chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();
        this.textToImageService = kernel.GetRequiredService<ITextToImageService>();
    }

    [KernelFunction]
    [Description("Creates an image from a text description")]
    public async Task<string> CreateImageFromTextAsync(string description, CancellationToken cancellationToken)
    {
        var imageTask = textToImageService.GenerateImageAsync(description, 1024, 1024, cancellationToken: cancellationToken);

        var systemPrompt = $@"Create a human response to indicate users that the image they requested has been created. The prompt for the image the user has requested is: {description}";

        var messageTask = chatCompletionService.GetChatMessageContentAsync(systemPrompt, new OpenAIPromptExecutionSettings()
        {
            MaxTokens = 50,
            Temperature = 1.0,
            TopP = 1.0,
        }, cancellationToken: cancellationToken);

        await Task.WhenAll(imageTask, messageTask);

        return $"{messageTask.Result.Content!} \n\n URL: {imageTask.Result}";
    }
}

//** OPTIONS **//

internal sealed class AzureOpenAIOptions
{
    /// <summary>
    /// Gets the Azure OpenAI API service version.
    /// </summary>
    public OpenAIClientOptions.ServiceVersion ServiceVersion { get; init; } = OpenAIClientOptions.ServiceVersion.V2024_02_15_Preview;

    /// <summary>
    /// Gets the <see cref="Uri "/> for an LLM resource (like OpenAI). This should include protocol and host name.
    /// </summary>
    [Required]
    [Uri]
    public required Uri Endpoint { get; init; }

    /// <summary>
    /// Gets the key credential used to authenticate to an LLM resource.
    /// </summary>
    [NotEmptyOrWhitespace]
    public required string Key { get; init; }

    /// <summary>
    /// Gets the model deployment name on the LLM (for example OpenAI) to use for chat.
    /// </summary>
    /// <remarks>
    /// WARNING: The model deployment name does not necessarily have to be the same as the model name. For example, a model of type `gpt-4` might be called «MyGPT»;
    /// this means that the value of this property does not necessarily indicate the model implemented behind it.
    /// Use property <see cref="ChatModelName"/>
    /// to set the model name.
    /// </remarks>
    [NotEmptyOrWhitespace]
    public required string ChatModelDeploymentName { get; init; }

    /// <summary>
    /// Gets the name (sort of a unique identifier) of the model to use for chat.
    /// </summary>
    /// <remarks>
    ///  This property is required if property <see cref="ChatModelDeploymentName"/> is not <see langword="null"/>.
    ///  It is usually used with the <see cref="Encamina.Enmarcha.AI.OpenAI.Abstractions.ModelInfo"/> class to get metadata and information about the model.
    ///  This model name must match the model names from the LLM (like OpenAI), like for example `gpt-4` or `gpt-35-turbo`.
    /// </remarks>
    [RequireWhenOtherPropertyNotNull(nameof(ChatModelDeploymentName))]
    [NotEmptyOrWhitespace]
    public required string ChatModelName { get; init; }

    /// <summary>
    /// Gets the model deployment name on the LLM (for example OpenAI) to use for chat.
    /// </summary>
    /// <remarks>
    /// <b>WARNING</b>: The model deployment name does not necessarily have to be the same as the model name. For example, a model of type `dall-e-3` might be called «MyDallE»;
    /// this means that the value of this property does not necessarily indicate the model implemented behind it.
    /// Use property <see cref="ImageGenerationModelName"/> to set the model name.
    /// </remarks>
    [NotEmptyOrWhitespace]
    public required string ImageGenerationModelDeploymentName { get; init; }

    /// <summary>
    /// Gets the name (sort of a unique identifier) of the model to use for chat.
    /// </summary>
    /// <remarks>
    /// This property is required if property <see cref="ImageGenerationModelDeploymentName"/> is not <see langword="null"/>.
    /// This model name must match the model names from the LLM (like OpenAI), like for example `dall-e-2` or `dall-e-3`.
    /// </remarks>
    [RequireWhenOtherPropertyNotNull(nameof(ImageGenerationModelDeploymentName))]
    [NotEmptyOrWhitespace]
    public required string ImageGenerationModelName { get; init; }

}

internal sealed class WeatherStackOptions
{
    /// <summary>
    /// Gets the WeatherStack API access key.
    /// </summary>
    [Required]
    public required string AccessKey { get; init; }
}

#pragma warning restore S3903 // Types should be defined in named namespaces

#pragma warning restore SKEXP0060
#pragma warning restore SKEXP0010
#pragma warning restore SKEXP0001
