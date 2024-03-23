//ref: https://note.com/nyaoki_board/n/na7c54c9ae2a5

import { app } from "/scripts/app.js";

app.registerExtension({
	name: "BatchString|cgem156",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BatchString|cgem156") {
			const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
			nodeType.prototype.getExtraMenuOptions = function (_, options) {
				const r = origGetExtraMenuOptions?.apply?.(this, arguments);
                    options.unshift(
                        {
                            content: "add input",
                            callback: () => {
                                var index = 1;
                                if (this.inputs != undefined){
                                    index += this.inputs.length;
                                }
                                this.addInput("text" + index, "STRING", {"multiline": true});
                            },
                        },
                        {
                            content: "remove input",
                            callback: () => {
                                if (this.inputs != undefined){
                                    this.removeInput(this.inputs.length - 1);
                                }								
                            },
                        },
                    );
                return r;
            }
        }
    },
});