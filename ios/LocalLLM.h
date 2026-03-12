#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

#ifdef RCT_NEW_ARCH_ENABLED
#import "LocalLLMSpec.h"
@interface LocalLLM : RCTEventEmitter <NativeLocalLLMSpec>
#else
@interface LocalLLM : RCTEventEmitter <RCTBridgeModule>
#endif
@end
